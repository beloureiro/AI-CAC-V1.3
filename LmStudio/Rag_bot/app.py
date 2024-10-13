# ---------------------------------------------------------------------
# Code to run the RAG bot with Streamlit final version and modularized
# --------------------------------------------------------------------- 
import streamlit as st
from ragbot import EnhancedRAGBot
from plot_utils import plot_similarity_scores, plot_confidence_gauge, get_confidence_legend
import logging
import numpy as np

@st.cache_resource
def get_ragbot():
    """Initializes and returns an instance of EnhancedRAGBot with database setup and model loading."""
    bot = EnhancedRAGBot()
    bot.initialize()
    return bot

def main():
    """Sets up the Streamlit interface for the Enhanced RAGBot Healthcare Coach app, handling user input, displaying chat messages, and visualizing analytics."""
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

                    # Display the response
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
    """Updates cache hit statistics, calculates the cache hit rate, and returns it."""
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
# streamlit run LmStudio/Rag_bot/app.py 
