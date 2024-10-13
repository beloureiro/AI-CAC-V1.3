# ragbot.py

import requests
import numpy as np
import datetime
import sqlite3
import logging
import re
import json
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from chromadb import Client
from rank_bm25 import BM25Okapi
from bert_score import score
import faiss

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class EnhancedRAGBot:
    CHAT_API_URL = "http://127.0.0.1:1234/v1/chat/completions"  # URL for chat
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    # Specialist mapping dictionary
    SPECIALIST_MAPPING = {
        'Sentiment_Patient_Experience_Expert': 'Patient Experience Expert',
        'Emotional_Intensity_Patient_Experience_Expert': 'Patient Experience Expert',
        'Urgency_Level_Patient_Experience_Expert': 'Patient Experience Expert',
        'Key_Issues_Patient_Experience_Expert': 'Patient Experience Expert',
        'Patient_Journey_Health_IT_Process_Expert': 'Health & IT Process Expert',
        'Positive_Aspects_Health_IT_Process_Expert': 'Health & IT Process Expert',
        'Inefficiencies_Health_IT_Process_Expert': 'Health & IT Process Expert',
        'Improvement_Suggestions_Health_IT_Process_Expert': 'Health & IT Process Expert',
        'Emotional_State_Clinical_Psychologist': 'Clinical Psychologist',
        'Support_Strategy_Clinical_Psychologist': 'Clinical Psychologist',
        'Suggested_Approach_Clinical_Psychologist': 'Clinical Psychologist',
        'Communication_Quality_Communication_Expert': 'Communication Expert',
        'Issues_Identified_Communication_Expert': 'Communication Expert',
        'Suggested_Improvements_Communication_Expert': 'Communication Expert',
        'Final_Recommendation_Communication_Expert': 'Communication Expert',
        'Key_Issues_Manager_and_Advisor': 'Manager and Advisor',
        'Recommendations_Manager_and_Advisor': 'Manager and Advisor',
    }

    # Source weights configuration
    SOURCE_WEIGHTS = {
        'patient_feedback': 1.0,  # Highest importance
        'patient_experience_expert': 1.0,
        'health_it_process_expert': 1.0,
        'clinical_psychologist': 1.0,
        'communication_expert': 1.0,
        'manager_and_advisor': 0.6,  # Lower importance
        'unknown_source': 0.5,  # Default weight for unknown sources
    }

    def __init__(self, db_path='LmStudio/Rag_bot/feedback_analysis.db'):
        self.db_path = db_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.initialize_db()
        self.initialization_done = False
        self.conversation_history = []
        self.tfidf_vectorizer = TfidfVectorizer()
        self.source_documents = {}
        self.bm25 = None
        self.st_model = self.model
        self.conversation_memory = []
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.client = Client()
        self.collection = self.client.get_or_create_collection("chat_history")

    def initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='aicac_consolidated'")
            if cursor.fetchone() is None:
                logging.error("Table 'aicac_consolidated' not found in the database.")
            else:
                logging.info(
                    "Successfully connected to the database and found 'aicac_consolidated' table.")

    def retrieve_similar_chunks(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str], List[float], List[float]]:
        try:
            if "provide" in query.lower() and "feedback" in query.lower():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT Patient_Feedback, Date FROM aicac_consolidated")
                    rows = cursor.fetchall()
                    feedbacks = [f"{row[0]} (Date: {row[1]})" for row in rows]
                    return feedbacks, ['Patient_Feedback'] * len(feedbacks), [1.0] * len(feedbacks), [1.0] * len(feedbacks)

            query_embedding = self.model.encode(query, convert_to_tensor=True)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM aicac_consolidated")
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]

                logging.debug(f"Columns: {columns}")
                logging.debug(f"Sample row: {rows[0] if rows else 'No rows'}")

                similarities = []
                current_date = datetime.datetime.now()
                for row in rows:
                    try:
                        feedback_date = datetime.datetime.strptime(row[1], '%d-%m-%Y')
                        age_in_years = (current_date - feedback_date).days / 365
                        time_decay_factor = max(0.5, 1 - (age_in_years / 2))

                        for idx, col_name in enumerate(columns):
                            if col_name.endswith('_embedding') or col_name in ['Feedback_ID', 'Date', 'Total_execution_time']:
                                continue

                            text_content = row[idx]
                            if not text_content:
                                continue

                            logging.debug(f"Processing column '{col_name}' at index {idx}")
                            logging.debug(f"Text content: {text_content}")

                            embedding_col_name = col_name + '_embedding'
                            if embedding_col_name in columns:
                                embedding_idx = columns.index(embedding_col_name)
                                embedding_data = row[embedding_idx]

                                if isinstance(embedding_data, bytes):
                                    row_embedding = np.frombuffer(embedding_data, dtype=np.float32)
                                else:
                                    row_embedding = np.array(embedding_data, dtype=np.float32)

                                if row_embedding.size == 0:
                                    continue

                                logging.debug(f"Row embedding shape: {row_embedding.shape}")
                            else:
                                continue

                            similarity = util.pytorch_cos_sim(query_embedding, row_embedding).item()
                            weighted_similarity = similarity * time_decay_factor
                            source_identifier = col_name

                            similarities.append((weighted_similarity, text_content, source_identifier))
                    except Exception as e:
                        logging.error(f"Error processing row: {e}")
                        continue

                similarities.sort(key=lambda x: x[0], reverse=True)
                top_items = similarities[:top_k]

                chunks = [item[1] for item in top_items]
                sources = [item[2] for item in top_items]
                distances = [1 - item[0] for item in top_items]
                similarities_scores = [item[0] for item in top_items]

                return chunks, sources, distances, similarities_scores
        except Exception as e:
            logging.error(f"Error retrieving chunks: {e}")
            return [], [], [], []

    def identify_source(self, source_identifier: str) -> str:
        source_mapping = {
            'Patient_Feedback': 'patient_feedback',
            'Sentiment_Patient_Experience_Expert': 'patient_experience_expert',
            'Emotional_Intensity_Patient_Experience_Expert': 'patient_experience_expert',
            'Urgency_Level_Patient_Experience_Expert': 'patient_experience_expert',
            'Key_Issues_Patient_Experience_Expert': 'patient_experience_expert',
            'Patient_Journey_Health_IT_Process_Expert': 'health_it_process_expert',
            'Positive_Aspects_Health_IT_Process_Expert': 'health_it_process_expert',
            'Inefficiencies_Health_IT_Process_Expert': 'health_it_process_expert',
            'Improvement_Suggestions_Health_IT_Process_Expert': 'health_it_process_expert',
            'Emotional_State_Clinical_Psychologist': 'clinical_psychologist',
            'Support_Strategy_Clinical_Psychologist': 'clinical_psychologist',
            'Suggested_Approach_Clinical_Psychologist': 'clinical_psychologist',
            'Communication_Quality_Communication_Expert': 'communication_expert',
            'Issues_Identified_Communication_Expert': 'communication_expert',
            'Suggested_Improvements_Communication_Expert': 'communication_expert',
            'Final_Recommendation_Communication_Expert': 'communication_expert',
            'Key_Issues_Manager_and_Advisor': 'manager_and_advisor',
            'Recommendations_Manager_and_Advisor': 'manager_and_advisor',
        }
        return source_mapping.get(source_identifier, 'unknown_source')

    def process_query(self, query: str) -> Tuple[str, List[str]]:
        try:
            relevant_chunks, sources, distances, similarities = self.retrieve_similar_chunks(query)
            response, _, _, _ = self.call_llm(query, relevant_chunks, sources, self.conversation_history)

            # Store the new response in memory
            self.add_to_memory(query, response)

            return response, []
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return self.fallback_response(query, [])

    def get_time_appropriate_greeting(self):
        current_hour = datetime.datetime.now().hour
        if 5 <= current_hour < 12:
            return "Good morning"
        elif 12 <= current_hour < 18:
            return "Good afternoon"
        else:
            return "Good evening"

    def calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        try:
            vectors = self.tfidf_vectorizer.fit_transform([text1, text2])
            return cosine_similarity(vectors[0], vectors[1])[0][0]
        except Exception as e:
            logging.error(f"Error in similarity calculation: {e}")
            return 0.0

    def verify_response(self, response: str, context: List[str]) -> Tuple[bool, float, List[str]]:
        if any(greeting in response.lower() for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return True, 1.0, []

        if not response or not context:
            return False, 0.0, []

        context_text = " ".join(context)
        similarity_score = self.calculate_similarity(response, context_text)

        verified = similarity_score > 0.4
        confidence = similarity_score

        source_docs = [filename for filename, content in self.source_documents.items()
                       if any(chunk in content for chunk in context)]

        return verified, confidence, source_docs

    def handle_greeting(self, query):
        greetings = ['hi', 'hello', 'hey', 'ola', 'olá', 'greetings']
        if query.lower().strip() in greetings:
            time_based_greeting = self.get_time_appropriate_greeting()
            return f"{time_based_greeting}! I'm your AI-Skills Advisor. How can I assist you with your healthcare professional development today?"
        return None

    def handle_identity_question(self, query):
        identity_questions = ['who are you', 'what are you', 'tell me about yourself', 'what can you do']
        if any(question in query.lower() for question in identity_questions):
            return (
                "I am the AI-Skills Advisor, a key component of the AI Clinical Advisory Crew. My role is to provide "
                "continuous, data-driven support to healthcare professionals like yourself. Here's what I can do for you:\n\n"
                "1. Analyze patient feedback and generate insights to improve care quality.\n"
                "2. Identify opportunities to enhance workflows and processes in healthcare delivery.\n"
                "3. Offer communication strategies to improve patient-provider interactions.\n"
                "4. Provide psychological insights for better post-consultation patient care.\n"
                "5. Deliver managerial overviews and summaries of patient feedback.\n"
                "6. Offer personalized recommendations for professional development.\n"
                "7. Provide instant, 24/7 access to AI-driven guidance and support.\n\n"
                "My goal is to help you excel in your healthcare practice by leveraging AI-powered insights "
                "and continuous learning. How can I assist you in improving your professional skills today?"
            )
        return None

    def add_to_memory(self, query: str, response: str):
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        self.collection.add_documents(embeddings=query_embedding, metadatas={
            "query": query, "response": response})

    def retrieve_from_memory(self, query: str, k: int = 3) -> List[Tuple[str, str]]:
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        results = self.collection.query(query_embedding, k=k)
        return [(doc['metadata']['query'], doc['metadata']['response']) for doc in results['documents']]

    def calculate_bert_score(self, response: str, context: str) -> float:
        P, R, F1 = score([response], [context], model_type='roberta-large')
        return F1.item()

    def calibrate_response(self, response: str, context: str) -> str:
        confidence = self.calculate_bert_score(response, context)
        if confidence > 0.8:
            return response
        elif confidence > 0.6:
            return f"According to my analysis, it is likely that {response}"
        else:
            return f"I'm not entirely certain, but it seems that {response}"

    def call_llm(self, query: str, context_chunks: List[str], sources: List[str], conversation_history: List[Dict[str, str]]) -> Tuple[str, bool, float, List[str]]:
        # Check for greeting first
        greeting_response = self.handle_greeting(query)
        if greeting_response:
            return greeting_response, True, 1.0, []

        # Check for identity question
        identity_response = self.handle_identity_question(query)
        if identity_response:
            return identity_response, True, 1.0, []

        # Handle feedback retrieval queries
        if "provide" in query.lower() and "feedback" in query.lower():
            feedback_response = self.process_feedback_query(query)
            return feedback_response, True, 1.0, []

        system_prompt = (
            "You are the AI-Skills Advisor, an AI assistant focused on healthcare professional development. "
            "Your purpose is to provide accurate and tailored responses to improve the skills of healthcare professionals. "
            "The user interacting with you is always a healthcare professional seeking advice to enhance their practice. "
            "When mentioning feedback or recommendations, always identify the specialist who provided it "
            "using the format 'According to [Specialist Name],'. "
            "Maintain this identity and provide responses without creating names or irrelevant details. "
            "Always frame your responses in the context of helping a healthcare professional improve their skills or practice."
        )

        user_prompt = self.build_user_prompt(query, context_chunks)

        payload = {
            "model": "meta-llama-3.1-8b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 500,
            "stream": False
        }

        logging.debug(f"LLM Request Payload: {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(self.CHAT_API_URL, json=payload, timeout=30)
            response.raise_for_status()
            response_content = response.json()['choices'][0]['message']['content']

            processed_response = self.process_llm_response(response_content, query)

            logging.debug(f"Processed LLM Response: {processed_response}")

            if self.st_model is None:
                self.st_model = self.model

            chunk_embeddings = self.st_model.encode(context_chunks)
            query_embedding = self.st_model.encode([query])[0]
            similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]

            weighted_similarities = []
            for i, (chunk, source_identifier) in enumerate(zip(context_chunks, sources)):
                source_key = self.identify_source(source_identifier)
                source_weight = self.SOURCE_WEIGHTS.get(source_key, 0.5)
                weighted_similarity = similarities[i] * source_weight
                weighted_similarities.append(weighted_similarity)

                logging.debug(f"Chunk {i+1}: Original similarity: {similarities[i]}, Source: {source_identifier}, Source key: {source_key}, Source weight: {source_weight}, Weighted similarity: {weighted_similarity}")

            final_confidence = np.mean(weighted_similarities)
            scaling_factor = 2.5  # Adjust based on desired amplification
            confidence_score = min(final_confidence * scaling_factor, 1.0)  # Ensure it doesn't exceed 1.0

            logging.debug(f"Final confidence: {final_confidence}, Scaled confidence: {confidence_score}")

            return processed_response, True, confidence_score, []
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            return self.fallback_response(query, context_chunks)

    def build_user_prompt(self, query: str, context_chunks: List[str]) -> str:
        if "provide" in query.lower() and "feedback" in query.lower():
            feedback_type = "all"
            if "positive" in query.lower():
                feedback_type = "positive"
            elif "negative" in query.lower():
                feedback_type = "negative"
            elif "neutral" in query.lower():
                feedback_type = "neutral"

            return (
                f"The user has asked: '{query}'. Respond as a chatbot and provide only the {feedback_type} patient feedback from the database. "
                f"Do not include the Feedback ID. Include the date for each feedback. "
                f"Do not add any analysis, greetings, or additional commentary. "
                f"List each feedback as a separate item. Use the following context information:\n\n"
                f"{self.get_context_info(context_chunks)}\n\n"
                f"Provide the requested patient feedback, exactly as given, without any introduction, conclusion, or additional analysis."
            )
        else:
            return (
                f"A healthcare professional has asked: '{query}'. Provide a concise response as an AI Healthcare Professional Coach, "
                f"focusing on their specific query and how it relates to their professional development or practice improvement. "
                f"Use the following context information:\n\n"
                f"{self.get_context_info(context_chunks)}\n\n"
                f"Maintain a supportive and professional tone, but avoid formal greetings or signatures. "
                f"Provide direct answers without unnecessary elaboration, always considering that you're advising a healthcare professional. "
                f"When mentioning feedback or recommendations, identify the specialist who provided it "
                f"using the format 'According to [Specialist Name],' based on the Specialist Information provided."
            )

    def get_context_info(self, context_chunks: List[str]) -> str:
        context_info = ""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM aicac_consolidated")
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            positive = []
            neutral = []
            negative = []
            specialist_info = {}

            for row in rows:
                feedback = f"{row[3]} (Date: {row[1]})"
                if row[4].lower() == 'positive':
                    positive.append(feedback)
                elif row[4].lower() == 'negative':
                    negative.append(feedback)
                else:
                    neutral.append(feedback)

                for i, value in enumerate(row):
                    if columns[i] in self.SPECIALIST_MAPPING and value:
                        specialist = self.SPECIALIST_MAPPING[columns[i]]
                        if specialist not in specialist_info:
                            specialist_info[specialist] = []
                        specialist_info[specialist].append(f"{columns[i]}: {value}")

            context_info += "Positive:\n" + "\n".join([f"- {f}" for f in positive]) + "\n\n"
            context_info += "Neutral:\n" + "\n".join([f"- {f}" for f in neutral]) + "\n\n"
            context_info += "Negative:\n" + "\n".join([f"- {f}" for f in negative]) + "\n\n"

            context_info += "Specialist Information:\n"
            for specialist, info in specialist_info.items():
                context_info += f"{specialist}:\n" + "\n".join([f"- {i}" for i in info]) + "\n\n"

        return context_info

    def is_greeting(self, query: str) -> bool:
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        return query.lower().strip() in greetings

    def is_identity_question(self, query: str) -> bool:
        identity_questions = ['who are you', 'what are you', 'tell me about yourself']
        return any(question in query.lower() for question in identity_questions)

    def fallback_response(self, query: str, context_chunks: List[str]) -> Tuple[str, bool, float, List[str]]:
        logging.info("Using fallback response mechanism")

        if self.is_greeting(query):
            time_based_greeting = self.get_time_appropriate_greeting()
            response = f"{time_based_greeting}! I'm your AI-Skills Advisor, part of the AI Clinical Advisory Crew. How can I assist you with your healthcare professional development today?"
        elif self.is_identity_question(query):
            response = (
                "I am the AI-Skills Advisor, a key component of the AI Clinical Advisory Crew. My role is to provide "
                "continuous, data-driven support to healthcare professionals like yourself. Here's what I can do for you:\n\n"
                "1. Analyze patient feedback and generate insights to improve care quality.\n"
                "2. Identify opportunities to enhance workflows and processes in healthcare delivery.\n"
                "3. Offer communication strategies to improve patient-provider interactions.\n"
                "4. Provide psychological insights for better post-consultation patient care.\n"
                "5. Deliver managerial overviews and summaries of patient feedback.\n"
                "6. Offer personalized recommendations for professional development.\n"
                "7. Provide instant, 24/7 access to AI-driven guidance and support.\n\n"
                "My goal is to help you excel in your healthcare practice by leveraging AI-powered insights "
                "and continuous learning. How can I assist you in improving your professional skills today?"
            )
        else:
            response = (
                "I apologize, but I'm currently experiencing some technical difficulties. As your AI-Skills Advisor, "
                "I'm here to help with your professional growth in healthcare. While I work on resolving this issue, "
                "could you please rephrase your query? I'm particularly equipped to assist with analyzing patient feedback, "
                "improving healthcare processes, enhancing communication strategies, and offering personalized professional "
                "development recommendations. What specific area of your healthcare practice would you like to focus on today?"
            )

        return response, True, 0.5, []

    def load_data(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT Patient_Feedback FROM aicac_consolidated")
            rows = cursor.fetchall()
            return " ".join([row[0] for row in rows if row[0]])

    def initialize(self):
        if not self.initialization_done:
            text = self.load_data()
            self.chunks = self.split_text_into_chunks(text)
            embeddings = self.generate_embeddings(self.chunks)
            if isinstance(embeddings[0], (list, np.ndarray)):
                dimension = len(embeddings[0])
            else:
                raise ValueError("Expected embeddings to be a list of vectors.")
            self.index = self.build_faiss_index(embeddings)
            self.bm25 = BM25Okapi(self.chunks)
            self.initialization_done = True

    def split_text_into_chunks(self, text, chunk_size=500):
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def build_faiss_index(self, embeddings):
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index

    def process_llm_response(self, response: str, query: str) -> str:
        logging.debug(f"Raw LLM response: {response}")

        response = re.sub(r'^(Dear.*?|Best regards.*?|AI-Skills.*?)(\n|$)',
                          '', response, flags=re.MULTILINE | re.IGNORECASE)
        response = re.sub(r'\n\s*\n', '\n\n', response)

        if "provide" in query.lower() and "feedback" in query.lower():
            feedback_items = re.findall(
                r'[-•]\s*(.*?)\s*\(Date:\s*(\d{2}-\d{2}-\d{4})\)', response, re.DOTALL | re.MULTILINE)

            logging.debug(f"Extracted feedback items: {feedback_items}")

            if feedback_items:
                feedback_type = "all"
                if "positive" in query.lower():
                    feedback_type = "Positive"
                elif "negative" in query.lower():
                    feedback_type = "Negative"
                elif "neutral" in query.lower():
                    feedback_type = "Neutral"

                if feedback_type != "all":
                    filtered_items = [
                        item for item in feedback_items if feedback_type.lower() in item[0].lower()]
                    if filtered_items:
                        return f"{feedback_type} Feedback:\n" + "\n".join([f"- *\"{item[0].strip()}\"* (Date: {item[1]})" for item in filtered_items])
                    else:
                        return f"No {feedback_type.lower()} patient feedback found in the database."
                else:
                    return f"All Feedback:\n" + "\n".join([f"- *\"{item[0].strip()}\"* (Date: {item[1]})" for item in feedback_items])
            else:
                return "No patient feedback found in the database."
        else:
            return response.strip()
