---
### What's New in Version 1.3
#### Introducing EnhancedRAGBot: A Data-Driven Healthcare Chatbot

Version 1.3 introduces **EnhancedRAGBot**, an advanced **Retrieval-Augmented Generation (RAG)** chatbot specifically designed to assist healthcare professionals. This new version leverages **SentenceTransformer (all-MiniLM-L6-v2)** for creating text embeddings and integrates with a **FAISS** index, which allows for efficient retrieval of contextually relevant expert feedback stored in a local database.

EnhancedRAGBot provides personalized coaching and real-time guidance by processing both structured and unstructured healthcare data, making it a powerful tool for healthcare providers seeking data-driven support.

#### Key Technical Features:
- **Real-Time Query Processing with Database-Driven Responses**: EnhancedRAGBot connects to a local SQLite database to retrieve patient feedback and relevant information. It offers real-time responses based on patient experiences and professional insights stored in the database.
- **Contextual Data Chunking and FAISS Indexing**: The chatbot uses FAISS to index text chunks from the database, enabling fast retrieval of contextually relevant feedback, which improves response precision.
- **Multi-Model and Multi-Step Similarity Assessment**: EnhancedRAGBot employs multiple similarity measures, such as TF-IDF, BM25, and cosine similarity with embeddings, ensuring that responses are highly relevant and contextually accurate.
- **Modular Caching and Efficient Data Retrieval**: By separating the stages of data retrieval, embedding generation, and model querying, EnhancedRAGBot achieves high maintainability. Caching mechanisms (`@st.cache_data` and `@st.cache_resource`) optimize performance, especially for repeated queries.
- **LLM API Integration with Configurable Parameters**: The bot connects to a large language model (LLM) via an API, providing fine-tuned responses based on user input. This setup allows the chatbot to tailor its responses, considering emotional cues and query context.
- **Performance Monitoring with Interactive Dashboard**: The system includes a dashboard with real-time analytics. Key metrics, like similarity scores and response confidence, are displayed in a user-friendly interface, helping users understand the relevance and reliability of responses.

#### How EnhancedRAGBot Works:
EnhancedRAGBot processes user queries by retrieving and analyzing data from a local SQLite database. It uses:
1. **Data Chunking**: Feedback is divided into manageable chunks, indexed with FAISS, and further processed using similarity metrics.
2. **Multi-Tiered Retrieval**: The bot integrates BM25 and TF-IDF algorithms for initial filtering, followed by cosine similarity for deeper contextual matching.
3. **Contextual Understanding and Real-Time LLM Interaction**: After retrieving relevant data chunks, EnhancedRAGBot constructs a user prompt and sends it to an LLM for final response generation, incorporating contextual understanding from its training.

#### Benefits of EnhancedRAGBot:
- **Data-Driven Insights**: Retrieves and analyzes structured feedback directly from the database, providing precise, actionable insights that support healthcare professional development.
- **Responsive and Contextually Accurate**: Multi-tiered similarity analysis ensures that each response is backed by relevant data, creating an experience that feels both personalized and well-informed.
- **Emotional Sensitivity in Responses**: The bot detects and adapts to emotional cues within queries, delivering a user experience that feels supportive and empathetic.
- **Scalable and Modular Architecture**: Modular design and caching enable easy updates and improved scalability, with a configuration adaptable to different healthcare settings.
- **Transparent Performance Metrics**: The interactive dashboard offers insight into the bot's response accuracy, allowing users to view similarity scores and assess the bot's performance in real-time.

---

# AI Clinical Advisory Crew (v1.3)

## Overview
The **AI Clinical Advisory Crew** is a sophisticated system aimed at transforming the healthcare experience by providing specialized AI-driven support. This setup involves a team of AI agents, each dedicated to a specific aspect of healthcare, such as patient feedback analysis, workflow optimization, emotional state assessment, and communication enhancement.

To explore how this system works, visit the live demo at [AI-CAC Viewer](https://ai-cac.streamlit.app/), or access the front-end repository [here](https://github.com/beloureiro/AI-CAC-Viewer.git).

This system provides **two main configurations** for AI agent operation:

1. **AI Agent Crew with the Ollama Framework**: This setup dynamically utilizes multiple large language models (LLMs) through the **CrewAI** framework, optimizing model selection (e.g., LLaMA, Hermes, Phi) for each task. This multi-agent approach enhances adaptability across diverse healthcare scenarios.
   
2. **Python-Only Local LLM Setup**: With **LM Studio**, users can run all operations locally, using models like **Meta-Llama-3.1-8B-Instruct-GGUF**. This setup ensures data privacy, allows for extensive control over model execution, and reduces costs by avoiding external APIs.

### Key System Benefits:
1. **Local LLM Support for Data Security**: Sensitive healthcare data remains internal.
2. **Cost-Effective**: No reliance on external APIs reduces operational expenses.
3. **Flexible Configurations**: Choose between multi-LLM setups with CrewAI or fully local execution with Python and LM Studio.
4. **Enhanced Control with LM Studio**: Adjust parameters like temperature and top_p while benefiting from GPU optimization for local execution.
5. **Streamlined Codebase in Version 1.3**: A clean, modular structure makes future updates and maintenance straightforward.

### LM Studio Integration Advantages
For local setups, **LM Studio** offers:

- **Concurrent Model Execution**: Run multiple models in parallel for direct comparison.
- **Fine-Grained Parameter Control**: Adjust settings like temperature and repetition penalty for precise outputs.
- **GPU Optimization**: Supports GPU use and model quantization for efficiency.
- **OpenAI-Compatible API**: Real-time results streaming.
- **Detailed Monitoring**: Track CPU/GPU metrics for effective resource management.

These features make LM Studio ideal for local tasks, enabling performance comparisons across models to choose the best configuration.

## AI Agents
Our team includes five specialized agents:
1. **Patient Experience Expert**: Analyzes patient feedback for key issues and emotional intensity, helping improve care quality.
2. **Health & IT Process Expert**: Maps patient journeys and identifies workflow inefficiencies, offering process improvement recommendations.
3. **Clinical Psychologist**: Assesses patient emotions, devising supportive strategies to enhance well-being.
4. **Communication Expert**: Evaluates provider-patient interactions, pinpointing areas for clearer and more empathetic communication.
5. **Manager and Advisor**: Synthesizes feedback, providing comprehensive reports and strategic recommendations.

Agents can be powered by a dynamic multi-LLM setup with CrewAI or a local configuration using LM Studio.

## Requirements
- Python 3.8+
- Install dependencies from `requirements.txt`

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

   ```

## Usage
Run the main script:
```
python main.py
```

The script reads patient feedback from input sources and performs analysis through the AI agent team.

## Project Structure
- `main.py`: Main script
- `config/`: Project configurations
- `agents/`: Agent definitions
- `tasks/`: Task definitions
- `utils/`: Utility functions

## Contributing
Contributions are welcome. Please open an issue to discuss proposed changes.

## License
[MIT](https://choosealicense.com/licenses/mit/)