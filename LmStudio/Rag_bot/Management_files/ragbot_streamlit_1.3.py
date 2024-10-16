# ----------------------------------------------
# Code to run the RAG bot with Streamlit v1.3
# ----------------------------------------------
import os
import requests
import numpy as np
import streamlit as st
import faiss
import plotly.graph_objects as go
from functools import lru_cache
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import re
import logging
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from chromadb import Client
from bert_score import score
import sqlite3

# Configuração de logging mais detalhada
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class EnhancedRAGBot:
    EMBEDDING_API_URL = "http://127.0.0.1:1234/v1/embeddings"  # URL para gerar embeddings
    CHAT_API_URL = "http://127.0.0.1:1234/v1/chat/completions"  # URL para chat
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    # Add the SPECIALIST_MAPPING dictionary
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
        'Recommendations_Manager_and_Advisor': 'Manager and Advisor'
    }

    """
    # Source Weight Configuration Guide
    This section defines the weights assigned to various data sources, which influence the confidence score of the assistant's responses. The `SOURCE_WEIGHTS` dictionary assigns a weight to each source type (based on column names from your database), reflecting its perceived reliability or importance. By adjusting these values, you can control the impact each source type has on the overall confidence calculation.
    ## How Source Weights Affect Confidence Score
    During response generation, each relevant text chunk retrieved from the database is associated with a `source_identifier`, which corresponds to the column name from which the text was extracted. The assistant calculates similarity scores between the user's query and these text chunks. Each similarity score is then multiplied by the weight assigned to its source in `SOURCE_WEIGHTS`. The weighted similarities are used to compute the final confidence score of the response.
    ## Adjusting Source Weights:
    - Increase Weight for High-Importance Sources: Boost the weight for sources you consider highly reliable or critical (e.g., increase from `0.8` to `1.0`) to emphasize their influence on the confidence score.
    - Reduce Weight for Lower-Importance Sources: Lower the weight for sources that are less reliable or less relevant (e.g., decrease from `0.7` to `0.5`) to reduce their impact on the confidence score.
    - Exclude a Source: Set the weight to `0.0` for sources you wish to exclude from influencing the confidence score entirely.
    - Equalize Weights: Set all sources to the same weight (e.g., `1.0`) to treat all sources equally, focusing solely on the similarity scores without considering source reliability.
    ## Interpreting Changes:
    - Higher Weights: Amplify the influence of that source type on the final confidence score, making responses based on that source more confident.
    - Lower Weights: Reduce the influence of that source type, allowing other sources to contribute more significantly to the confidence score.
    - Zero Weight: Effectively removes the source from contributing to the confidence score, disregarding its content in the final response.
    Adjust these weights based on the specific needs of your application to reflect the importance or reliability of each source type in your confidence calculation.
    """

    SOURCE_WEIGHTS = {
        'patient_feedback': 1.0,                 # Highest importance
        'patient_experience_expert': 1.0,
        'health_it_process_expert': 1.0,
        'clinical_psychologist': 1.0,
        'communication_expert': 1.0,
        'manager_and_advisor': 0.6,              # Lower importance
        'unknown_source': 0.5,                   # Default weight for unknown sources
    }


    def __init__(self, db_path='LmStudio/Rag_bot/feedback_analysis.db'):
        self.db_path = db_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.st_model = self.model  # Adicione esta linha
        self.initialize_db()
        # Inicializa a variável para controle de inicialização
        self.initialization_done = False
        self.conversation_history = []
        self.tfidf_vectorizer = TfidfVectorizer()  # Inicializa o vetor TF-IDF
        self.source_documents = {}  # Dicionário para armazenar documentos de origem
        self.bm25 = None  # Adiciona BM25 para recuperação
        self.st_model = None  # Adiciona modelo de SentenceTransformer
        self.conversation_memory = []  # Memória de conversa
        # Load Cross-Encoder for ranking
        self.cross_encoder = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.client = Client()  # Ajuste a inicialização do client
        self.collection = self.client.get_or_create_collection(
            "chat_history")  # Create collection for chat history

    def initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='aicac_consolidated'")
            if cursor.fetchone() is None:
                print("Table 'aicac_consolidated' not found in the database.")
            else:
                print(
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
            relevant_chunks, distances, similarities = self.retrieve_similar_chunks(
                query)
            response, _, _, _ = self.call_llm(query, relevant_chunks, [])

            # Store the new response in memory
            self.add_to_memory(query, response)

            return response, []
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return self.fallback_response(query, [])

    # def check_emotional_cues(self, query):
    #     positive_cues = ['happy', 'satisfied',
    #                      'pleased', 'grateful', 'excited']
    #     negative_cues = ['upset', 'frustrated',
    #                      'angry', 'disappointed', 'worried']

    #     words = query.lower().split()
    #     positive_match = any(cue in words for cue in positive_cues)
    #     negative_match = any(cue in words for cue in negative_cues)

    #     if positive_match and not negative_match:
    #         return "positive"
    #     elif negative_match and not positive_match:
    #         return "negative"
    #     else:
    #         return "neutral"

    def summarize_expert_feedback(self):
        text = self.load_data()
        summary = {expert: {"positive": [], "improvement": []} for expert in ["PatientExperienceExpert",
                                                                              "HealthITProcessExpert", "ClinicalPsychologist", "CommunicationExpert", "ManagerAndAdvisor"]}

        for expert in summary.keys():
            if expert in text:
                expert_sections = text.split(expert)[1].split('\n\n')
                for section in expert_sections:
                    if section.strip():
                        if "improvement" in section.lower() or "issue" in section.lower():
                            summary[expert]["improvement"].append(
                                section.strip())
                        else:
                            summary[expert]["positive"].append(section.strip())

        return summary

    def extract_patient_feedback(self):
        text = self.load_data()
        patient_feedback = {
            "positive": [],
            "negative": []
        }

        feedback_start = text.find("Feedback:")
        if feedback_start != -1:
            feedback_text = text[feedback_start:].split('\n', 1)[1].strip()
            patient_feedback["positive"].append(feedback_text)

        return patient_feedback

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
            vectors = self.tfidf_vectorizer.fit_transform(
                [text1, text2])  # Calcula a similaridade usando TF-IDF
            return cosine_similarity(vectors[0], vectors[1])[0][0]
        except Exception as e:
            logging.error(f"Error in similarity calculation: {e}")
            return 0.0

    def verify_response(self, response: str, context: List[str]) -> Tuple[bool, float, List[str]]:
        # Simplificado para não depender do fact_checker
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
        identity_questions = ['who are you', 'what are you',
                              'tell me about yourself', 'what can you do']
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

    def extract_patient_feedbacks(self, chunks: List[str]) -> List[Dict[str, str]]:
        feedbacks = []
        for chunk in chunks:
            feedback_matches = re.finditer(
                r"Patient Feedback:(.*?)(?=Patient Feedback:|$)", chunk, re.DOTALL)
            for match in feedback_matches:
                feedback_text = match.group(1).strip()
                date_match = re.search(
                    r"Date of the Patient feedback: (\d{2}-\d{2}-\d{4})", feedback_text)

                if feedback_text and date_match:
                    feedbacks.append({
                        "text": feedback_text,
                        "date": date_match.group(1)
                    })
        return feedbacks

    def get_feedback_by_query(self, query: str) -> str:
        # Extrai data ou palavras-chave específicas da consulta
        date_match = re.search(r'\d{2}-\d{2}-\d{4}', query)
        keyword = re.search(r'\b(positive|negative|neutral)\b', query.lower())

        feedback_chunks = []

        if date_match:
            date_str = date_match.group(0)
            for chunk in self.chunks:
                if date_str in chunk:
                    feedback_chunks.append(chunk)
        elif keyword:
            keyword_str = keyword.group(0)
            for chunk in self.chunks:
                if keyword_str in chunk.lower():
                    feedback_chunks.append(chunk)
        else:
            # Nenhuma data ou palavra-chave especificada, retorna todos os feedbacks
            feedback_chunks = self.chunks

        return "\n".join(feedback_chunks)


    def process_feedback_query(self, query: str) -> str:
        feedback_response = self.get_feedback_by_query(query)
        if feedback_response:
            return feedback_response

    def add_to_memory(self, query: str, response: str):
        query_embedding = self.model.encode(
            [query], convert_to_tensor=False)  # Get embedding for query
        self.collection.add_documents(embeddings=query_embedding, metadatas={
                                      "query": query, "response": response})  # Store in ChromaDB

    def retrieve_from_memory(self, query: str, k: int = 3) -> List[Tuple[str, str]]:
        query_embedding = self.model.encode(
            [query], convert_to_tensor=False)  # Get embedding for query
        results = self.collection.query(query_embedding, k=k)  # Query ChromaDB
        # Return past conversations
        return [(doc['metadata']['query'], doc['metadata']['response']) for doc in results['documents']]

    def calculate_bert_score(self, response: str, context: str) -> float:
        # Calculate BERTScore
        P, R, F1 = score([response], [context], model_type='roberta-large')
        return F1.item()

    def calibrate_response(self, response: str, context: str) -> str:
        confidence = self.calculate_bert_score(
            response, context)  # Get confidence score
        if confidence > 0.8:
            return response
        elif confidence > 0.6:
            return f"According to my analysis, it is likely that {response}"
        else:
            return f"I'm not entirely certain, but it seems that {response}"

    def generate_response(self, query: str) -> str:
        # Retrieve and rank relevant documents
        relevant_docs = self.retrieve_and_rank(query)
        context = " ".join(relevant_docs)  # Combine context
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        generated_text = self.generator_pipeline(
            input_text, max_length=200, num_return_sequences=1)  # Generate response
        return generated_text[0]['generated_text']

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
                f"The user has asked: '{query}'. Respond as a chatbot and provide only the {
                    feedback_type} patient feedback from the database. "
                f"Do not include the Feedback ID. Include the date for each feedback. "
                f"Do not add any analysis, greetings, or additional commentary. "
                f"List each feedback as a separate item. Use the following context information:\n\n"
                f"{self.get_context_info(context_chunks)}\n\n"
                f"Provide the requested patient feedback, exactly as given, without any introduction, conclusion, or additional analysis."
            )
        else:
            return (
                f"A healthcare professional has asked: '{
                    query}'. Provide a concise response as an AI Healthcare Professional Coach, "
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
                        specialist_info[specialist].append(
                            f"{columns[i]}: {value}")

            context_info += "Positive:\n" + \
                "\n".join([f"- {f}" for f in positive]) + "\n\n"
            context_info += "Neutral:\n" + \
                "\n".join([f"- {f}" for f in neutral]) + "\n\n"
            context_info += "Negative:\n" + \
                "\n".join([f"- {f}" for f in negative]) + "\n\n"

            context_info += "Specialist Information:\n"
            for specialist, info in specialist_info.items():
                context_info += f"{specialist}:\n" + \
                    "\n".join([f"- {i}" for i in info]) + "\n\n"

        return context_info

    def is_greeting(self, query: str) -> bool:
        greetings = ['hi', 'hello', 'hey', 'good morning',
                     'good afternoon', 'good evening']
        return query.lower().strip() in greetings

    def is_identity_question(self, query: str) -> bool:
        identity_questions = ['who are you',
                              'what are you', 'tell me about yourself']
        return any(question in query.lower() for question in identity_questions)

    def fallback_response(self, query: str, context_chunks: List[str]) -> Tuple[str, bool, float, List[str]]:
        logging.info("Using fallback response mechanism")

        if self.is_greeting(query):
            time_based_greeting = self.get_time_appropriate_greeting()
            response = f"{
                time_based_greeting}! I'm your AI-Skills Advisor, part of the AI Clinical Advisory Crew. How can I assist you with your healthcare professional development today?"
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

        # Use a default confidence of 0.5 for fallback responses
        return response, True, 0.5, []

    def regenerate_response(self, query: str, context_chunks: List[str], conversation_history: List[Dict[str, str]]) -> str:
        try:
            relevant_context = self.get_relevant_context(query, context_chunks)
            conversation_context = self.format_conversation_history(
                conversation_history)

            stricter_prompt = (
                f"Based on the following context and conversation history, provide a concise and relevant answer to the user's query. "
                f"Context: {relevant_context}\n"
                f"Conversation History: {conversation_context}\n"
                f"Query: {query}\n"
                "Respond as an AI Healthcare Professional Coach, focusing on healthcare professional development. "
                "If the query is off-topic, politely guide the conversation back to relevant topics."
            )

            payload = {
                "model": "meta-llama-3.1-8b-instruct",
                "messages": [
                    {"role": "system", "content": "You are an AI Healthcare Professional Coach. Provide concise, relevant, and helpful responses."},
                    {"role": "user", "content": stricter_prompt}
                ],
                "temperature": 0.2,  # Temperatura mais baixa para respostas mais conservadoras
                "max_tokens": 300,   # Ajuste de max_tokens
                "stream": False
            }

            response = requests.post(self.LM_STUDIO_API_URL, json=payload)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return "Error: Failed to regenerate response."
        except Exception as e:
            logging.error(f"Error in regenerate_response: {e}")
            return "Error: Failed to regenerate response due to an unexpected error."

    def get_relevant_context(self, query: str, context_chunks: List[str]) -> str:
        # Use TF-IDF para encontrar os chunks mais relevantes
        tfidf = TfidfVectorizer().fit_transform([query] + context_chunks)
        similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        # Pegar os 3 chunks mais relevantes
        top_indices = similarities.argsort()[-3:][::-1]

        relevant_context = "\n".join([context_chunks[i] for i in top_indices])
        return relevant_context

    def format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        formatted_history = ""
        # Limitar a 5 mensagens mais recentes
        for message in conversation_history[-5:]:
            role = "User" if message["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {message['content']}\n"
        return formatted_history

    def load_data(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT Patient_Feedback FROM aicac_consolidated")
            rows = cursor.fetchall()
            return " ".join([row[0] for row in rows if row[0]])

    def initialize(self):
        if not self.initialization_done:
            text = self.load_data()
            st.success(f"Successfully loaded data from the database")
            self.chunks = self.split_text_into_chunks(text)
            embeddings = self.generate_embeddings(
                self.chunks)  # Passa todos os chunks de uma vez
            if isinstance(embeddings[0], (list, np.ndarray)):
                dimension = len(embeddings[0])
            else:
                raise ValueError(
                    "Expected embeddings to be a list of vectors.")
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

    # @lru_cache(maxsize=1000)
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def build_faiss_index(self, embeddings):
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index

    def calibrate_confidence(self, response: str, confidence: float) -> str:
        if confidence > 0.8:
            return response
        elif confidence > 0.6:
            return f"Based on the available information, I believe that {response}"
        elif confidence > 0.4:
            return f"While I'm not entirely certain, my understanding is that {response}"
        else:
            return f"I'm not very confident about this, but here's what I can say: {response}"

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
                # If no feedback items were found, return a message indicating no feedback
                return "No patient feedback found in the database."
        else:
            # For non-feedback queries, return the processed response
            return response.strip()
        
    def extract_date(self, item: str) -> str:
        date_match = re.search(r'\(Date:\s*(\d{2}-\d{2}-\d{4})\)', item)
        return date_match.group(1) if date_match else "Unknown Date"


def plot_similarity_scores(chunks, distances):
    similarity_scores = 1 / (1 + np.array(distances))
    max_score = max(similarity_scores)

    fig = go.Figure(data=[go.Bar(
        x=[f"Chunk {i+1}" for i in range(len(chunks))],
        y=similarity_scores,
        text=[f"{score:.2f}" for score in similarity_scores],
        textposition='auto',
        marker_color='rgb(53, 133, 93)',
    )])

    fig.update_layout(
        yaxis_title="Similarity Score",
        yaxis_range=[0, max_score * 1.1],  # Dynamic scale
        template="plotly_dark",
        height=150,  # Reduzir a altura do gráfico
        margin=dict(l=0, r=10, t=10, b=0),  # Reduzir as margens
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def plot_confidence_gauge(confidence: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,  # Converting to percentage
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Response Confidence", 'font': {'size': 16}},
        number={'font': {'size': 30}, 'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'size': 14}},
            'bar': {'color': "darkgreen"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "rgba(255,255,255,0.1)"},
                {'range': [50, 80], 'color': "rgba(255,255,255,0.3)"},
                {'range': [80, 100], 'color': "rgba(255,255,255,0.5)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'size': 14}
    )

    return fig


def get_confidence_legend(confidence: float) -> str:
    if confidence >= 0.7:
        return "High confidence: The response is likely accurate and well-supported by the data."
    elif confidence >= 0.5:
        return "Moderate confidence: The response is generally reliable but may have some uncertainties."
    elif confidence >= 0.3:
        return "Low confidence: The response may be speculative or based on limited information."
    else:
        return "Very low confidence: The response should be treated as highly uncertain."


@st.cache_resource
def get_ragbot():
    bot = EnhancedRAGBot()
    bot.initialize()
    return bot


def main():
    st.set_page_config(page_title="Enhanced RAGBot Healthcare Coach",
                       layout="wide", initial_sidebar_state="expanded")
    st.title("AI Skills Advisor with Enhanced Verification")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #0e1525;
        }
        .stTextInput > div > div > input {
            background-color: #0e1525;
            color: white;
        }
        .stTextInput > div > div > input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }
        .stTextInput > div > div {
            border-color: #1e2a3a;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    ragbot = get_ragbot()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "cache_hits" not in st.session_state:
        st.session_state.cache_hits = 0
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

    user_avatar_path = "LmStudio/Rag_bot/assets/profile-picture.png"
    assistant_avatar_path = "LmStudio/Rag_bot/assets/neural.png"

    for message in st.session_state.messages:
        avatar_path = user_avatar_path if message["role"] == "user" else assistant_avatar_path
        with st.chat_message(message["role"], avatar=avatar_path):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your healthcare coach a question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=user_avatar_path):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=assistant_avatar_path):
            message_placeholder = st.empty()
            with st.spinner("Processing your query..."):
                try:
                    st.session_state.total_queries += 1
                    relevant_chunks, sources, distances, similarities = ragbot.retrieve_similar_chunks(prompt)
                    response, verified, confidence, source_docs = ragbot.call_llm(prompt, relevant_chunks, sources, st.session_state.messages)

                    logging.debug(f"Relevant chunks: {relevant_chunks}")
                    logging.debug(f"Sources: {sources}")
                    logging.debug(f"Distances: {distances}")
                    logging.debug(f"Similarities: {similarities}")

                    verification_status = "✅ Verified" if verified else "⚠️ Unverified"
                    full_response = f"{verification_status} Response (Confidence: {confidence * 100:.2f}%)\n\n{response}"

                    # Inside the main() function, replace the existing code that constructs full_response with the following:

                    if "provide" in prompt.lower() and "feedback" in prompt.lower():
                        full_response = f"{verification_status} Response (Confidence: {confidence * 100:.2f}%)"
                        feedback_count = len(relevant_chunks)
                        feedback_type = "all"
                        if "positive" in prompt.lower():
                            feedback_type = "positive"
                        elif "negative" in prompt.lower():
                            feedback_type = "negative"
                        elif "neutral" in prompt.lower():
                            feedback_type = "neutral"

                        if feedback_count > 0:
                            full_response += f"\n\nFound {feedback_count} {feedback_type} patient feedback(s):"
                            for feedback in relevant_chunks:
                                full_response += f"\n- {feedback}"
                        else:
                            full_response += f"\n\nNo {feedback_type} patient feedback found in the database."
                    else:
                        full_response = f"{verification_status} Response (Confidence: {confidence * 100:.2f}%)\n\n{response}"

                    if source_docs:
                        full_response += "\n\n**Sources:**"
                        for doc in source_docs:
                            full_response += f"\n- {doc}"

                    message_placeholder.markdown(full_response)


                except Exception as e:
                    logging.error(f"Error processing query: {e}")
                    response, verified, confidence, source_docs = ragbot.fallback_response(prompt, [])
                    distances = []
                    similarities = []
                    verification_status = "⚠️ Fallback Response"
                    full_response = f"{verification_status} (Confidence: {confidence * 100:.2f}%)\n\n{response}"
                    message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        with st.sidebar:
            st.markdown(
                "<h3 style='text-align: center; font-size: 16px;'>RAGBot Analytics</h3>", unsafe_allow_html=True)

            if 'distances' in locals() and distances:
                fig_confidence = plot_confidence_gauge(confidence)
                st.plotly_chart(fig_confidence, use_container_width=True, config={
                                'displayModeBar': False})

                legend_text = get_confidence_legend(confidence)
                st.markdown(f"<span style='color: #35855d; font-size: 14px;'>{legend_text}</span>", unsafe_allow_html=True)

            with st.container():
                if 'distances' in locals() and distances:
                    similarities = 1 / (1 + np.array(distances))
                    fig_similarity = plot_similarity_scores(
                        relevant_chunks, similarities)
                    st.plotly_chart(fig_similarity, use_container_width=True)
                    st.markdown(
                        "<span style='color: #35855d; font-size: 14px;'>This chart shows how similar each retrieved chunk is to your query. "
                        "Higher bars indicate greater relevance to your question.</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning(
                        "No similarity scores available for this query.")


            with st.expander("View Relevant Context", expanded=False):
                st.markdown(
                    "This section shows the most relevant text chunks used to answer your query. "
                    "Each chunk is ranked by its similarity to your question."
                )
                if 'distances' in locals() and distances:
                    for i, (chunk, distance) in enumerate(zip(relevant_chunks, distances), 1):
                        similarity_score = 1 / (1 + distance)
                        st.markdown(
                            f"**Chunk {i}:** (Similarity Score: {similarity_score:.2f})")
                        st.write(chunk)
                        st.markdown("---")
                else:
                    st.warning("No relevant context available for this query.")


def update_cache_statistics():
    if "cache_hits" not in st.session_state:
        st.session_state.cache_hits = 0
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

    st.session_state.total_queries += 1

    cache_hit_rate = (st.session_state.cache_hits / st.session_state.total_queries) * \
        100 if st.session_state.total_queries > 0 else 0

    return cache_hit_rate


if __name__ == "__main__":
    main()

# to run the app, type in the terminal:
# streamlit run LmStudio/Rag_bot/ragbot_streamlit_1.3.py
