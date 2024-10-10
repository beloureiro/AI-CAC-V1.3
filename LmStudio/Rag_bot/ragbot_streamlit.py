import os
import requests
import numpy as np
import streamlit as st
import faiss
import plotly.graph_objects as go
from functools import lru_cache
import datetime
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import re
import logging
import json

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        # Inicialização do fact checker com tratamento de erros
        try:
            self.fact_checker = pipeline("zero-shot-classification", 
                                         model="facebook/bart-large-mnli", 
                                         device=-1)  # Use CPU
        except Exception as e:
            logging.error(f"Failed to initialize fact checker: {e}")
            self.fact_checker = None

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
        query_embedding = self.generate_embeddings([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.chunks[i] for i in indices[0]], distances[0].tolist()  # Retorna distâncias como lista

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

    def check_factual_consistency(self, response: str, context: str) -> float:
        if self.fact_checker is None or not response or not context:
            logging.warning("Skipping factual consistency check due to missing components.")
            return 0.5  # Valor neutro

        try:
            input_text = f"premise: {context} hypothesis: {response}"
            result = self.fact_checker(input_text, candidate_labels=["entailment", "contradiction"], multi_label=False)
            entailment_score = next((item['score'] for item in result if item['label'] == 'entailment'), 0.0)
            return entailment_score
        except Exception as e:
            logging.error(f"Error in factual consistency check: {e}")
            return 0.5  # Valor neutro em caso de erro

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
        if any(greeting in response.lower() for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return True, 1.0, []

        if not response or not context:
            return False, 0.0, []

        context_text = " ".join(context)
        factual_score = self.check_factual_consistency(response, context_text)
        similarity_score = self.calculate_similarity(response, context_text)
        
        verified = factual_score > 0.6 and similarity_score > 0.4  # Ajuste de thresholds
        confidence = (factual_score + similarity_score) / 2
        
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
                sentiment_match = re.search(r"Sentiment_Patient_Experience_Expert: (\w+)", feedback_text)
                
                if feedback_text and date_match:
                    sentiment = sentiment_match.group(1) if sentiment_match else "Not specified"
                    feedbacks.append({
                        "text": feedback_text,
                        "date": date_match.group(1),
                        "sentiment": sentiment
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
            response += f"   Sentiment: {feedback['sentiment']}\n"
            response += f"   {feedback['text']}\n\n"
        return response

    def call_llm(self, query: str, context_chunks: List[str], conversation_history: List[Dict[str, str]]) -> Tuple[str, bool, float, List[str]]:
        feedbacks = self.extract_patient_feedbacks(context_chunks)
        feedback_summary = json.dumps(feedbacks, indent=2)

        prompt = (
            f"You are an AI Healthcare Professional Coach, part of the AI Clinical Advisory Crew. "
            f"Your role is to provide guidance and support for healthcare professional development. "
            f"Here are all the patient feedbacks extracted from the database:\n{feedback_summary}\n\n"
            f"When responding to queries about patient feedbacks, always include ALL feedbacks in your response. "
            f"Summarize the feedbacks with their dates and sentiments. Include the total number of feedbacks and highlight any trends or notable issues. "
            f"Context:\n{' '.join(context_chunks)}\n\n"
            f"Conversation History:\n{self.format_conversation_history(conversation_history)}\n\n"
            f"User Query: {query}\n\n"
            "Please provide a comprehensive and relevant response to the user's query, staying in character as the AI Healthcare Professional Coach."
        )

        payload = {
            "model": "meta-llama-3.1-8b-instruct",
            "messages": [
                {"role": "system", "content": "You are an AI Healthcare Professional Coach. Provide comprehensive, relevant, and helpful responses."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000,  # Aumentado para permitir respostas mais completas
            "stream": False
        }

        try:
            response = requests.post(self.CHAT_API_URL, json=payload, timeout=30)
            response.raise_for_status()
            response_content = response.json()['choices'][0]['message']['content']
            return response_content, True, 1.0, [f"Feedback_{i+1}" for i in range(len(feedbacks))]
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            return self.fallback_response(query, context_chunks)

    def fallback_response(self, query: str, context_chunks: List[str]) -> Tuple[str, bool, float, List[str]]:
        greeting_response = self.handle_greeting(query)
        if greeting_response:
            return greeting_response, True, 1.0, []

        logging.info("Using fallback response mechanism")
        if query.lower() in ['hi', 'hello', 'hey']:
            response = "Hello! I'm your AI Healthcare Professional Coach. How can I assist you with your healthcare professional development today?"
        elif 'who are you' in query.lower():
            response = "I am an AI Healthcare Professional Coach, designed to assist with various aspects of healthcare professional development. I represent a team of AI experts including Patient Experience, Health IT Process, Clinical Psychology, Communication, and Management. How can I help you today?"
        else:
            response = (
                "I apologize, but I'm currently experiencing some technical difficulties. "
                "As your AI Healthcare Professional Coach, I'm here to assist you with healthcare professional development. "
                "Could you please rephrase your question or provide more details about what you'd like to know? "
                "I'll do my best to help you based on the information available to me."
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
        margin=dict(l=10, r=10, t=30, b=10),  # Reduzir as margens
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


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
                    relevant_chunks, distances = ragbot.retrieve_similar_chunks(prompt)  # Captura distâncias
                    response, verified, confidence, source_docs = ragbot.call_llm(prompt, relevant_chunks, st.session_state.messages)
                except Exception as e:
                    logging.error(f"Error processing query: {e}")
                    response, verified, confidence, source_docs = ragbot.fallback_response(prompt, [])
                    distances = []  # Inicializa distances como lista vazia

            if verified:
                st.markdown(f"✅ Verified Response (Confidence: {confidence:.2f})")
            else:
                st.markdown(f"⚠️ Unverified Response (Confidence: {confidence:.2f})")
            
            st.markdown(response)
            
            if source_docs:
                st.markdown("**Sources:**")
                for doc in source_docs:
                    st.markdown(f"- {doc}")

        st.session_state.messages.append(
            {"role": "assistant", "content": response})

        with st.sidebar:
            # st.sidebar.markdown("""
            #     <div style='background-color: #0e1117; padding: 5px; border-radius: 10px;'>
            #         <h2 style="text-align: center; font-size: 21px;">
            #             <span style="color: #1b9e4b; font-style: italic;">AI</span> 
            #             <span style="color: white;">Clinical Advisory</span> 
            #             <span style="color: #1b9e4b; font-style: italic;">Crew</span>
            #         </h2>
            #     </div>
            # """, unsafe_allow_html=True)
            st.markdown(
                """
                <h3 style='text-align: center;'>RAGBot Analytics</h3>
                """,
                unsafe_allow_html=True
            )
            with st.container():
                if distances:  # Verifica se há distâncias para plotar
                    fig = plot_similarity_scores(relevant_chunks, distances)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(
                        "<span style='color: #35855d; font-size: 14px;'>This chart shows how similar each retrieved chunk is to your query. "
                        "Higher bars indicate greater relevance to your question.</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("No similarity scores available for this query.")

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

            with st.expander("View Relevant Context", expanded=False):
                st.markdown(
                    "This section shows the most relevant text chunks used to answer your query. "
                    "Each chunk is ranked by its similarity to your question."
                )
                if distances:  # Verifica se há distâncias para exibir
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