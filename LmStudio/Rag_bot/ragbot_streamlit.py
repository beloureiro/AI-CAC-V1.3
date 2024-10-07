import os
import requests
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import plotly.graph_objects as go
from functools import lru_cache


class RAGBot:
    LM_STUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"

    def __init__(self, data_directory=None, consolidated_path='consolidated_text.txt', chunk_size=500):
        if data_directory is None:
            data_directory = os.path.join(
                os.path.dirname(__file__), '..', 'lms_reports_md')

        self.data_directory = os.path.abspath(data_directory)
        self.consolidated_path = os.path.join(
            os.path.dirname(__file__), consolidated_path)
        self.chunk_size = chunk_size
        self.model = None
        self.index = None
        self.chunks = []
        self.conversation_history = []

    @st.cache_data
    def load_data(_self):
        all_text = ""
        success_message = ""
        try:
            if os.path.exists(_self.consolidated_path) and os.path.getsize(_self.consolidated_path) > 0:
                with open(_self.consolidated_path, 'r', encoding='utf-8') as file:
                    all_text = file.read()
                success_message = f"Successfully loaded data from {
                    _self.consolidated_path}"
            else:
                raise FileNotFoundError(
                    f"File {_self.consolidated_path} not found or is empty.")
        except FileNotFoundError as e:
            all_text = "This is a fallback text for demonstration purposes. The actual consolidated text file could not be loaded."
            success_message = f"Warning: {str(e)} Using fallback text."
        except Exception as e:
            all_text = "Error occurred. Using fallback text for demonstration."
            success_message = f"Error: An unexpected error occurred while loading the file: {
                str(e)}"

        if not all_text:
            all_text = "Fallback text for empty file scenario."
            success_message = "Warning: The loaded text is empty. Using fallback text."

        return all_text, success_message

    @st.cache_data
    def split_text_into_chunks(_self, text):
        return [text[i:i+_self.chunk_size] for i in range(0, len(text), _self.chunk_size)]

    @st.cache_resource
    def load_model(_self):
        return SentenceTransformer('all-MiniLM-L6-v2')

    @st.cache_data
    def generate_embeddings(_self, chunks):
        model = _self.load_model()
        return model.encode(chunks)

    @st.cache_resource
    def build_faiss_index(_self, embeddings):
        embedding_dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(embeddings))
        return index

    @lru_cache(maxsize=100)
    def retrieve_similar_chunks(_self, query, _index, _chunks):
        model = _self.load_model()
        query_embedding = model.encode([query])[0]
        distances, indices = _index.search(np.array([query_embedding]), 5)
        return [_chunks[i] for i in indices[0]], distances[0]

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
        text, _ = self.load_data()
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
        text, _ = self.load_data()
        patient_feedback = {
            "positive": [],
            "negative": []
        }

        feedback_start = text.find("Patient Feedback:")
        if feedback_start != -1:
            feedback_text = text[feedback_start:]
            feedback_lines = feedback_text.split('\n')

            current_category = None
            current_feedback = ""
            for line in feedback_lines:
                if line.strip().lower().startswith("positive feedback:"):
                    if current_feedback and current_category:
                        patient_feedback[current_category].append(
                            current_feedback.strip())
                    current_category = "positive"
                    current_feedback = ""
                elif line.strip().lower().startswith("negative feedback:"):
                    if current_feedback and current_category:
                        patient_feedback[current_category].append(
                            current_feedback.strip())
                    current_category = "negative"
                    current_feedback = ""
                elif line.strip() and current_category:
                    current_feedback += " " + line.strip()

            if current_feedback and current_category:
                patient_feedback[current_category].append(
                    current_feedback.strip())

        return patient_feedback

    @st.cache_data
    def call_llm(_self, query, context_chunks, conversation_history, structured=False):
        system_message = {
            "role": "system",
            "content": (
                "You are 'your healthcare professional coach' representing a team of experts including "
                "PatientExperienceExpert, HealthITProcessExpert, ClinicalPsychologist, CommunicationExpert, and ManagerAndAdvisor. "
                "Provide brief, direct, and actionable advice based on the collective feedback from these experts. "
                "Address the user's specific questions without repeating them. "
                "Offer clear, factual responses and professional guidance based only on the information given. "
                "Ensure you accurately reflect both positive feedback and areas for improvement. "
                "If clarification is needed, ask a short, direct question."
            )
        }

        expert_summary = _self.summarize_expert_feedback()
        patient_feedback = _self.extract_patient_feedback()

        expert_areas = {
            "communication": "CommunicationExpert",
            "process": "HealthITProcessExpert",
            "patient experience": "PatientExperienceExpert",
            "psychology": "ClinicalPsychologist",
            "management": "ManagerAndAdvisor"
        }

        filtered_expert_summary = expert_summary
        for area, expert in expert_areas.items():
            if area in query.lower():
                filtered_expert_summary = {expert: expert_summary[expert]}
                break

        context_combined = "Expert Feedback Summary:\n"
        for expert, feedback in filtered_expert_summary.items():
            context_combined += f"{expert}:\n"
            context_combined += "  Positive: " + \
                "; ".join(feedback["positive"]) + "\n"
            context_combined += "  Areas for Improvement: " + \
                "; ".join(feedback["improvement"]) + "\n\n"

        context_combined += "Patient Feedback:\n"
        context_combined += "  Positive:\n" + \
            "\n".join(
                [f"    - {feedback}" for feedback in patient_feedback["positive"]]) + "\n"
        context_combined += "  Negative:\n" + \
            "\n".join(
                [f"    - {feedback}" for feedback in patient_feedback["negative"]]) + "\n\n"

        context_combined += f"Additional Context:\n" + \
            "\n".join(context_chunks)

        emotional_state = _self.check_emotional_cues(query)

        history_context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_history[-10:]])

        if query.lower().strip().split()[0] in ['hello', 'hi', 'hey', 'greetings']:
            prompt = f"The user said: {query}. Respond with a professional greeting and ask how you can assist with their healthcare professional development today. Mention that you represent a team of experts including PatientExperienceExpert, HealthITProcessExpert, ClinicalPsychologist, CommunicationExpert, and ManagerAndAdvisor."
        elif any(area in query.lower() for area in expert_areas.keys()) or "improve" in query.lower() or "guidance" in query.lower():
            prompt = (
                f"Context:\n{context_combined}\n\nConversation History:\n{
                    history_context}\n\nQuestion: {query}\n"
                "Provide a comprehensive list of areas for improvement based on the relevant expert feedback. "
                "Structure the response by expert, clearly attributing each point to its source. "
                "Include ALL points mentioned by the relevant expert(s), even if they are minor. "
                "Present the list in a clear, bullet-point format, with brief explanations for each point. "
                "After listing all improvements, summarize the key areas that need immediate attention."
            )
        elif "strengths" in query.lower() or "positive" in query.lower():
            prompt = (
                f"Context:\n{context_combined}\n\nConversation History:\n{
                    history_context}\n\nQuestion: {query}\n"
                "Based on the experts' feedback and previous conversation, highlight the strengths of the professional performance. "
                "Emphasize factual observations and successful actions without assuming the user's emotional state."
            )
        elif "who" in query.lower() and ("told" in query.lower() or "said" in query.lower()):
            prompt = (
                f"Context:\n{context_combined}\n\nConversation History:\n{
                    history_context}\n\nQuestion: {query}\n"
                "Provide a detailed summary of what each expert said about the user's services. "
                "Include specific feedback from PatientExperienceExpert, HealthITProcessExpert, "
                "ClinicalPsychologist, CommunicationExpert, and ManagerAndAdvisor. "
                "Present the information in a clear, structured format."
            )
        elif "feedback" in query.lower() or "patient" in query.lower():
            prompt = (
                f"Context:\n{context_combined}\n\nConversation History:\n{
                    history_context}\n\nQuestion: {query}\n"
                "Provide only the patient feedback, both positive and negative, in full without truncation. "
                "Do not include any expert evaluations or analysis in this response. "
                "Use the exact words from the patients' feedback. "
                "Clearly separate and label positive and negative feedback. "
                "If there are multiple pieces of feedback, number them for clarity. "
                "Do not summarize or paraphrase the feedback; present it as it appears in the context."
            )
        else:
            prompt = (
                f"Context:\n{context_combined}\n\nConversation History:\n{
                    history_context}\n\nQuestion: {query}\n"
                "Provide a helpful response based on the given context and previous conversation. If the query is off-topic, "
                "politely guide the conversation back to healthcare professional development. "
                "Avoid making assumptions about the user's feelings or satisfaction level."
            )

        if emotional_state != "neutral":
            prompt += f"\nNote: The user's query suggests a {
                emotional_state} emotional state. Consider this in your response, but avoid making assumptions. If necessary, ask for clarification about their feelings."

        prompt += " Respond concisely in 2-3 sentences unless asked for detailed feedback."

        payload = {
            "model": "meta-llama-3.1-8b-instruct",
            "messages": [
                system_message,
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 400,
            "stream": False
        }

        if structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "coaching_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "improvement_suggestions": {"type": "string"},
                            "strengths": {"type": "string"},
                            "feedback_source": {"type": "string"}
                        },
                        "required": ["improvement_suggestions", "strengths"]
                    }
                }
            }

        response = requests.post(_self.LM_STUDIO_API_URL, json=payload)

        if response.status_code == 200:
            response_content = response.json(
            )['choices'][0]['message']['content']
            if "improve" in query.lower() or "guidance" in query.lower() or "feedback" in query.lower() or "patient" in query.lower():
                return response_content
            else:
                sentences = response_content.split('.')
                truncated_response = '. '.join(
                    sentences[:3]) + ('.' if len(sentences) > 3 else '')
                return truncated_response
        else:
            return "Error: Failed to get response from LM Studio."

    def initialize(self):
        text, success_message = self.load_data()
        st.success(success_message)
        self.chunks = self.split_text_into_chunks(text)
        embeddings = self.generate_embeddings(self.chunks)
        self.index = self.build_faiss_index(embeddings)


def plot_similarity_scores(chunks, distances):
    similarity_scores = 1 / (1 + np.array(distances))

    fig = go.Figure(data=[go.Bar(
        x=[f"Chunk {i+1}" for i in range(len(chunks))],
        y=similarity_scores,
        text=[f"{score:.2f}" for score in similarity_scores],
        textposition='auto',
        marker_color='rgb(53, 133, 93)',
    )])

    fig.update_layout(
        title="Similarity Scores for Retrieved Chunks",
        # xaxis_title="Chunks",
        yaxis_title="Similarity Score",
        yaxis_range=[0, 1],
        template="plotly_dark",
        height=400,
    )

    return fig


@st.cache_resource
def get_ragbot():
    bot = RAGBot()
    bot.initialize()
    return bot


def main():
    st.set_page_config(page_title="RAGBot Healthcare Coach", layout="wide")
    st.title("RAGBot Healthcare Professional Coach")

    st.markdown(
        '''
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .sidebar .sidebar-content .block-container {
            padding-top: 0rem;
            padding-bottom: 2rem;
        }
        .sidebar .sidebar-content .stExpander {
            background-color: white;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

    ragbot = get_ragbot()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "cache_hits" not in st.session_state:
        st.session_state.cache_hits = 0

    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your healthcare coach a question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                st.session_state.total_queries += 1
                relevant_chunks, distances = ragbot.retrieve_similar_chunks(
                    prompt, ragbot.index, tuple(ragbot.chunks))
                if ragbot.retrieve_similar_chunks.cache_info().hits > st.session_state.cache_hits:
                    st.session_state.cache_hits += 1
                response = ragbot.call_llm(
                    prompt, relevant_chunks, st.session_state.messages)
            st.markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})

        with st.sidebar:
            st.markdown(
                """
                <h1 style='text-align: center;'>RAGBot Analytics</h1>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")
            with st.container():
                # st.write("Similarity Scores Visualization")
                fig = plot_similarity_scores(relevant_chunks, distances)
                st.plotly_chart(fig)
                st.markdown(
                    "This chart shows how similar each retrieved chunk is to your query. "
                    "Higher bars indicate greater relevance to your question."
                )

            with st.expander("Cache Statistics"):
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

            with st.expander("View Relevant Context"):
                st.markdown(
                    "This section shows the most relevant text chunks used to answer your query. "
                    "Each chunk is ranked by its similarity to your question."
                )
                for i, (chunk, distance) in enumerate(zip(relevant_chunks, distances), 1):
                    similarity_score = 1 / (1 + distance)
                    st.markdown(
                        f"**Chunk {i}:** (Similarity Score: {similarity_score:.2f})")
                    st.write(chunk)
                    st.markdown("---")


if __name__ == "__main__":
    main()

# to run the app, type in the terminal:
# streamlit run LmStudio/Rag_bot/ragbot_streamlit.py

