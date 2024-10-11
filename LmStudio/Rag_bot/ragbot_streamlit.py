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
from sentence_transformers import SentenceTransformer

# Configuração de logging mais detalhada
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedRAGBot:
    EMBEDDING_API_URL = "http://127.0.0.1:1234/v1/embeddings"  # URL para gerar embeddings
    CHAT_API_URL = "http://127.0.0.1:1234/v1/chat/completions"  # URL para chat
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(self, data_directory=None, consolidated_path='consolidated_text.txt', chunk_size=500):
        if data_directory is None:
            data_directory = os.path.join(
                os.path.dirname(__file__), '..', 'lms_reports_md')

        self.data_directory = os.path.abspath(data_directory)
        self.consolidated_path = os.path.join(
            os.path.dirname(__file__), consolidated_path)
        self.chunk_size = chunk_size
        self.index = None
        self.chunks = []
        self.conversation_history = []
        self.initialization_done = False
        self.tfidf_vectorizer = TfidfVectorizer()  # Inicializa o vetor TF-IDF
        self.source_documents = {}  # Dicionário para armazenar documentos de origem
        self.bm25 = None  # Adiciona BM25 para recuperação
        self.st_model = None  # Adiciona modelo de SentenceTransformer
        self.conversation_memory = []  # Memória de conversa

    def consolidate_md_files(self):
        all_text = ""
        for filename in os.listdir(self.data_directory):
            if filename.endswith(".md"):
                file_path = os.path.join(self.data_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_text += file.read() + "\n"  # Add a newline between files for clarity

        if all_text:
            with open(self.consolidated_path, 'w', encoding='utf-8') as outfile:
                outfile.write(all_text)
            print(f"Consolidated text saved to {self.consolidated_path}")
        else:
            print("No .md files found. consolidated_text.txt was not updated.")

    @st.cache_data
    def load_data(_self):
        _self.consolidate_md_files()  # Call the consolidation method before loading
        try:
            if os.path.exists(_self.consolidated_path) and os.path.getsize(_self.consolidated_path) > 0:
                with open(_self.consolidated_path, 'r', encoding='utf-8') as file:
                    all_text = file.read()
            else:
                raise FileNotFoundError(f"File {_self.consolidated_path} not found or is empty.")
        except FileNotFoundError as e:
            all_text = "This is a fallback text for demonstration purposes. The actual consolidated text file could not be loaded."
        except Exception as e:
            all_text = "Error occurred. Using fallback text for demonstration."

        if not all_text:
            all_text = "Fallback text for empty file scenario."

        _self.source_documents = _self._load_source_documents()  # Carrega documentos de origem

        return all_text

    def _load_source_documents(_self) -> Dict[str, str]:
        documents = {}
        for filename in os.listdir(_self.data_directory):
            if filename.endswith(".md"):
                file_path = os.path.join(_self.data_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents[filename] = file.read()  # Carrega o conteúdo dos documentos
        return documents

    @st.cache_data
    def split_text_into_chunks(_self, text):
        # Implementa uma estratégia de divisão de texto melhorada
        sentences = re.split(r'(?<=[.!?]) +', text)  # Divide o texto em sentenças
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= _self.chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "nomic-embed-text-v1.5",
            "input": texts
        }

        try:
            response = requests.post(self.EMBEDDING_API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return []

    @st.cache_resource
    def build_faiss_index(_self, embeddings):
        embedding_dim = len(embeddings[0])
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(embeddings))
        return index

    def retrieve_similar_chunks(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        # BM25 retrieval
        bm25_scores = self.bm25.get_scores(query)
        bm25_top_k = np.argsort(bm25_scores)[-top_k:][::-1]

        # Dense retrieval
        query_embedding = self.st_model.encode([query])[0]
        chunk_embeddings = self.st_model.encode(self.chunks)
        dense_scores = np.dot(chunk_embeddings, query_embedding)
        dense_top_k = np.argsort(dense_scores)[-top_k:][::-1]

        # Combine results
        combined_indices = list(set(bm25_top_k) | set(dense_top_k))
        combined_scores = [max(bm25_scores[i], dense_scores[i]) for i in combined_indices]
        
        top_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_indices], [combined_scores[i] for i in top_indices]

    def check_emotional_cues(self, query):
        positive_cues = ['happy', 'satisfied',
                         'pleased', 'grateful', 'excited']
        negative_cues = ['upset', 'frustrated',
                         'angry', 'disappointed', 'worried']

        words = query.lower().split()
        positive_match = any(cue in words for cue in positive_cues)
        negative_match = any(cue in words for cue in negative_cues)

        if positive_match and not negative_match:
            return "positive"
        elif negative_match and not positive_match:
            return "negative"
        else:
            return "neutral"

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
            vectors = self.tfidf_vectorizer.fit_transform([text1, text2])  # Calcula a similaridade usando TF-IDF
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
            return (
                f"{self.get_time_appropriate_greeting()}! I'm your AI Healthcare Professional Coach "
                f"from the AI Clinical Advisory Crew. How can I assist you with your healthcare professional development today?"
            )
        return None

    def handle_identity_question(self, query):
        identity_questions = ['who are you', 'what are you', 'tell me about yourself']
        if any(question in query.lower() for question in identity_questions):
            return (
                "I am an AI Healthcare Professional Coach, part of the AI Clinical Advisory Crew. "
                "My role is to assist healthcare professionals like you in various aspects of your practice, "
                "including patient experience, health IT processes, clinical psychology, communication, and management. "
                "I can provide insights, analysis, and recommendations to help you improve your professional skills and patient care. "
                "How can I help you enhance your healthcare practice today?"
            )
        return None

    def extract_patient_feedbacks(self, chunks: List[str]) -> List[Dict[str, str]]:
        feedbacks = []
        for chunk in chunks:
            feedback_matches = re.finditer(r"Patient Feedback:(.*?)(?=Patient Feedback:|$)", chunk, re.DOTALL)
            for match in feedback_matches:
                feedback_text = match.group(1).strip()
                date_match = re.search(r"Date of the Patient feedback: (\d{2}-\d{2}-\d{4})", feedback_text)
                
                if feedback_text and date_match:
                    feedbacks.append({
                        "text": feedback_text,
                        "date": date_match.group(1)
                    })
        return feedbacks

    def process_feedback_query(self, query: str) -> str:
        feedbacks = self.extract_patient_feedbacks(self.chunks)
        
        if "raw" in query.lower() or "text" in query.lower():
            return "\n\n".join([f"{feedback['text']}" for feedback in feedbacks])
        
        if "summary" in query.lower():
            return self.summarize_feedbacks(feedbacks)
        
        if "sentiments" in query.lower():
            return self.get_feedback_sentiments(feedbacks)
        
        # Default: retorna todos os feedbacks com detalhes
        response = "Here are all the patient feedbacks from our database:\n\n"
        for i, feedback in enumerate(feedbacks, 1):
            response += f"{i}. Date: {feedback['date']}\n"
            response += f"   {feedback['text']}\n\n"
        return response

    def add_to_memory(self, text: str, role: str):
        self.conversation_memory.append({"text": text, "role": role})
        if len(self.conversation_memory) > 10:  # Mantenha apenas as últimas 10 interações
            self.conversation_memory.pop(0)

    def get_conversation_context(self) -> str:
        return "\n".join([f"{item['role']}: {item['text']}" for item in self.conversation_memory])

    def call_llm(self, query: str, context_chunks: List[str], conversation_history: List[Dict[str, str]]) -> Tuple[str, bool, float, List[str]]:
        system_prompt = (
            "You are the AI-Skills Advisor, an advanced AI coach for healthcare professionals. "
            "Your role is to provide personalized guidance, support professional development, "
            "and offer data-driven insights to improve healthcare delivery. "
            "Maintain a professional and supportive tone throughout the conversation."
        )

        conversation_context = self.get_conversation_context()
        
        # Construa o prompt do usuário com base no tipo de consulta
        user_prompt = self.build_user_prompt(query, conversation_context, context_chunks)

        payload = {
            "model": "meta-llama-3.1-8b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
            "stream": False
        }

        logging.debug(f"LLM Request Payload: {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(self.CHAT_API_URL, json=payload, timeout=30)
            response.raise_for_status()
            response_content = response.json()['choices'][0]['message']['content']
            logging.debug(f"LLM Response: {response_content}")

            # Calcular a confiança baseada na similaridade dos chunks
            chunk_embeddings = self.st_model.encode(context_chunks)
            query_embedding = self.st_model.encode([query])[0]
            similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
            
            # Return similarities instead of confidence
            return response_content, True, similarities, []
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            return self.fallback_response(query, context_chunks)

    def build_user_prompt(self, query: str, conversation_context: str, context_chunks: List[str]) -> str:
        if self.is_greeting(query):
            return (
                f"Conversation history:\n{conversation_context}\n\n"
                f"The user has greeted you with '{query}'. Respond with a warm, professional greeting "
                f"and briefly introduce yourself as the AI-Skills Advisor. Ask how you can assist them "
                f"with their professional development in healthcare today."
            )
        elif self.is_identity_question(query):
            return (
                f"Conversation history:\n{conversation_context}\n\n"
                f"The user has asked '{query}'. Provide a concise explanation of your role as the AI-Skills Advisor. "
                f"Emphasize your ability to support healthcare professionals in their development and "
                f"offer to assist with any specific areas they'd like to focus on."
            )
        elif "patient feedback" in query.lower():
            feedbacks = self.extract_patient_feedbacks(self.chunks)
            feedback_summary = json.dumps(feedbacks, indent=2)
            return (
                f"Conversation history:\n{conversation_context}\n\n"
                f"The user has requested patient feedback. Here are all the patient feedbacks extracted from the database:\n{feedback_summary}\n\n"
                f"Provide a concise summary of the feedback, including the number of feedbacks, dates, and key points. "
                f"Do not include any additional analysis or comments."
            )
        else:
            return (
                f"Conversation history:\n{conversation_context}\n\n"
                f"The user has asked: '{query}'. Provide a response that focuses on their professional growth "
                f"and addresses their specific query. If relevant, incorporate insights from the following context: "
                f"{' '.join(context_chunks)}\n\n"
                f"Maintain a supportive and professional tone, offering guidance and resources tailored to healthcare professionals."
            )

    def is_greeting(self, query: str) -> bool:
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        return query.lower().strip() in greetings

    def is_identity_question(self, query: str) -> bool:
        identity_questions = ['who are you', 'what are you', 'tell me about yourself']
        return any(question in query.lower() for question in identity_questions)

    def fallback_response(self, query: str, context_chunks: List[str]) -> Tuple[str, bool, float, List[str]]:
        logging.info("Using fallback response mechanism")
        if query.lower() in ['hi', 'hello', 'hey']:
            response = "Hello! I'm your AI-Skills Advisor. How can I assist you with your professional development in healthcare today?"
        else:
            response = (
                "I apologize for the inconvenience. As your AI-Skills Advisor, I'm here to support your growth as a healthcare professional. "
                "Could you please rephrase your question or share more about what aspect of your professional development you'd like to focus on? "
                "I'm here to provide guidance, whether it's about improving patient care, enhancing communication skills, or exploring new medical knowledge."
            )
        return response, True, 1.0, []

    def regenerate_response(self, query: str, context_chunks: List[str], conversation_history: List[Dict[str, str]]) -> str:
        try:
            relevant_context = self.get_relevant_context(query, context_chunks)
            conversation_context = self.format_conversation_history(conversation_history)
            
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
        top_indices = similarities.argsort()[-3:][::-1]  # Pegar os 3 chunks mais relevantes
        
        relevant_context = "\n".join([context_chunks[i] for i in top_indices])
        return relevant_context

    def format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        formatted_history = ""
        for message in conversation_history[-5:]:  # Limitar a 5 mensagens mais recentes
            role = "User" if message["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {message['content']}\n"
        return formatted_history

    def initialize(self):
        text = self.load_data()
        if not hasattr(self, 'initialization_done'):
            st.success(f"Successfully loaded data from {self.consolidated_path}")
            self.initialization_done = True
        self.chunks = self.split_text_into_chunks(text)
        embeddings = self.generate_embeddings(self.chunks)
        self.index = self.build_faiss_index(embeddings)
        self.bm25 = BM25Okapi(self.chunks)  # Inicializa BM25
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')  # Inicializa modelo de SentenceTransformer

    def process_query(self, query: str) -> Tuple[str, List[str]]:
        """Process the user's query to find relevant chunks and generate a response."""
        # Generate embeddings for the query
        query_embedding = self.generate_embeddings([query])
        if not query_embedding:
            return "Error generating embeddings for the query.", []

        # Retrieve similar chunks using the FAISS index
        if self.index is None:
            return "Index not initialized.", []

        distances, indices = self.index.search(np.array(query_embedding), 5)  # Retrieve top 5 similar chunks
        relevant_chunks = [self.chunks[i] for i in indices[0]]

        # Here you can implement logic to generate a response based on the relevant chunks
        response = self.generate_response(relevant_chunks)

        return response, relevant_chunks

    def generate_response(self, relevant_chunks: List[str]) -> str:
        """Generate a response based on the relevant chunks."""
        # Combine relevant chunks into a single response
        response = "Here are the relevant pieces of information:\n"
        for chunk in relevant_chunks:
            response += f"- {chunk}\n"
        return response

    def calibrate_confidence(self, response: str, confidence: float) -> str:
        if confidence > 0.8:
            return response
        elif confidence > 0.6:
            return f"Based on the available information, I believe that {response}"
        elif confidence > 0.4:
            return f"While I'm not entirely certain, my understanding is that {response}"
        else:
            return f"I'm not very confident about this, but here's what I can say: {response}"


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
    if confidence >= 0.8:
        return "High confidence: The response is likely accurate and well-supported by the data."
    elif confidence >= 0.6:
        return "Moderate confidence: The response is generally reliable but may have some uncertainties."
    elif confidence >= 0.4:
        return "Low confidence: The response may be speculative or based on limited information."
    else:
        return "Very low confidence: The response should be treated as highly uncertain."


@st.cache_resource
def get_ragbot():
    bot = EnhancedRAGBot()
    bot.initialize()
    return bot


def main():
    st.set_page_config(page_title="Enhanced RAGBot Healthcare Coach", layout="wide", initial_sidebar_state="expanded")
    st.title("AI Skills Advisor with Enhanced Verification")

    st.markdown(
        """
        <style>
        /* Custom sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #0e1525;
        }
        
        /* Custom chat input box color */
        .stTextInput > div > div > input {
            background-color: #0e1525;
            color: white;
        }
        
        /* Custom chat input box placeholder color */
        .stTextInput > div > div > input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }
        
        /* Custom chat input box border color */
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

    # Define o caminho para os avatares
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
            with st.spinner("Processing your query..."):
                try:
                    st.session_state.total_queries += 1
                    relevant_chunks, distances = ragbot.retrieve_similar_chunks(prompt)
                    response, verified, similarities, source_docs = ragbot.call_llm(prompt, relevant_chunks, st.session_state.messages)
                    
                    # Log de debug para visualizar chunks relevantes e distâncias
                    logging.debug(f"Relevant chunks: {relevant_chunks}")
                    logging.debug(f"Distances: {distances}")
                    
                except Exception as e:
                    logging.error(f"Error processing query: {e}")
                    response, verified, similarities, source_docs = ragbot.fallback_response(prompt, [])
                    distances = []

            confidence = 1 / (1 + np.mean(similarities))  # Calculate confidence using distances
            st.markdown(f"✅ Verified Response (Confidence: {confidence * 100:.2f}%)")
            
            st.markdown(response)
            
            if source_docs:
                st.markdown("**Sources:**")
                for doc in source_docs:
                    st.markdown(f"- {doc}")

        st.session_state.messages.append(
            {"role": "assistant", "content": response})

        with st.sidebar:
            # Mantenha o cabeçalho existente
            st.markdown(
                """
                <h3 style='text-align: center; font-size: 16px;'>RAGBot Analytics</h3>
                """,
                unsafe_allow_html=True
            )

            # Adicione o novo gráfico de confiança
            if 'distances' in locals() and distances:
                fig_confidence = plot_confidence_gauge(confidence)
                st.plotly_chart(fig_confidence, use_container_width=True, config={'displayModeBar': False})
                
                # Add the legend as a separate Streamlit element
                legend_text = get_confidence_legend(confidence)
                st.markdown(f"<span style='color: #35855d; font-size: 14px;'>{legend_text}</span>", unsafe_allow_html=True)
                
                #st.markdown(f"<span style='color: #35855d; font-size: 10px;'>Confidence: {confidence:.2f}</span>", unsafe_allow_html=True)

            # Mantenha o gráfico de similaridade existente
            with st.container():
                if 'distances' in locals() and distances:
                    fig_similarity = plot_similarity_scores(relevant_chunks, distances)
                    st.plotly_chart(fig_similarity, use_container_width=True)
                    st.markdown(
                        "<span style='color: #35855d; font-size: 14px;'>This chart shows how similar each retrieved chunk is to your query. "
                        "Higher bars indicate greater relevance to your question.</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("No similarity scores available for this query.")

            # Mantenha as estatísticas de cache existentes
            with st.expander("Cache Statistics", expanded=True):
                cache_hit_rate = (st.session_state.cache_hits / st.session_state.total_queries) * \
                    100 if st.session_state.total_queries > 0 else 0
                st.write(f"Cache Hit Rate: {cache_hit_rate:.2f}%")
                st.markdown(
                    "<small>Percentage of queries answered using cached results. "
                    "A higher rate means faster responses for repeated questions.</small>",
                    unsafe_allow_html=True
                )
                st.write(f"Total Queries: {st.session_state.total_queries}")
                st.markdown(
                    "<small>Total number of questions you've asked.</small>", unsafe_allow_html=True)
                st.write(f"Cache Hits: {st.session_state.cache_hits}")
                st.markdown(
                    "<small>Number of times a query was answered using cached results, "
                    "resulting in faster response times.</small>",
                    unsafe_allow_html=True
                )
                st.write(f"Cache Misses: {
                         st.session_state.total_queries - st.session_state.cache_hits}")
                st.markdown(
                    "<small>Number of times a query required new processing. "
                    "These queries may take longer to answer.</small>",
                    unsafe_allow_html=True
                )

            # Mantenha a exibição do contexto relevante
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


if __name__ == "__main__":
    main()

# to run the app, type in the terminal:
# streamlit run LmStudio/Rag_bot/ragbot_streamlit.py