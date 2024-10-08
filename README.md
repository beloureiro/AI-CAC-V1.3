---
### What's New in Version 1.3
#### Introducing RAGBot: A Healthcare Coaching Assistant

Version 1.3 introduces **RAGBot**, a **Retrieval-Augmented Generation (RAG)** chatbot designed to support healthcare professionals. This feature integrates **SentenceTransformer (all-MiniLM-L6-v2)** for generating text embeddings with a **FAISS** index for efficient retrieval of contextually relevant expert feedback. RAGBot functions as a personalized coaching assistant, offering real-time guidance based on consolidated data.

#### Key Technical Features:
- **New Real-Time Chat Functionality**: RAGBot provides direct, real-time interaction, allowing users to pose questions and receive contextually relevant feedback that is driven by expert insights. The chatbot tailors responses based on the context and emotional cues detected within the queries.
- **Chunked Data Processing and FAISS Indexing**: Text data is consolidated from markdown files, split into fixed-size chunks, and indexed using FAISS, enabling quick retrieval of relevant text segments. This method enhances response accuracy by focusing on the most contextually similar chunks.
- **Modular Architecture with Multi-level Caching**: The architecture separates data processing, embedding generation, and FAISS indexing, making the system highly maintainable. Additionally, caching mechanisms (using `@st.cache_data` and `@st.cache_resource`) are employed across various components—data loading, embedding generation, and model loading—accelerating responses, particularly for repeated queries.
- **API-based LLM Integration with Configurable Parameters**: Through a custom API, RAGBot interacts with a large language model (LLM), allowing fine-tuning of response characteristics, including temperature and token limits. This provides precision in responses, tailoring advice according to user queries and emotional cues.
- **Comprehensive Performance Dashboard**: The sidebar contains a real-time dashboard displaying analytics on RAGBot’s performance. Key metrics include cache hit rates, total query counts, and similarity scores for retrieved chunks. This offers users detailed insights into system efficiency and response accuracy.

#### Benefits of RAGBot:
- **Contextualized Expert-Driven Assistance**: By using contextually relevant text chunks, RAGBot provides detailed, actionable insights tailored to specific domains like patient experience, process improvement, and communication, thereby enhancing professional development.
- **Optimized Performance Through Caching and Indexing**: Chunked text data indexed with FAISS, combined with multi-level caching, ensures rapid retrieval of information and reduced latency in responses, improving the user experience and facilitating quick access to expert knowledge.
- **Personalized Interaction with Emotional Awareness**: The chatbot detects emotional cues in user input, adjusting responses to better support and engage the user, offering an interaction experience that feels personalized and empathetic.
- **Enhanced Maintainability and Scalability**: The modular design and separation of processes allow easy updates and expansions. The API-driven architecture is also configurable, enabling adaptation to a wide range of healthcare settings.
- **Insightful Real-Time Performance Monitoring**: The sidebar dashboard provides metrics like cache hit rates, total query counts, and similarity scores, allowing users to monitor system efficiency and understand the relevance of responses. Users can view individual chunks retrieved with similarity scores, offering transparency and the ability to gauge the system's contextual accuracy.

This version significantly extends RAGBot’s functionality with new real-time, context-aware interactions, and adds a comprehensive dashboard that empowers healthcare professionals by delivering tailored support and operational transparency.


---
# AI Clinical Advisory Crew (v1.3)

## Description
Welcome to the **AI Clinical Advisory Crew**, an advanced and flexible system designed to transform and elevate the patient experience in healthcare. This project brings together a team of specialized AI agents, each with a unique role in analyzing patient feedback, improving healthcare workflows, assessing emotional states, and delivering actionable recommendations for communication and operational improvements.

For a live demonstration of how this system works in practice, visit the frontend Viewer platform that visualizes AI agent outputs: [AI-CAC Viewer](https://ai-cac.streamlit.app/). You can also explore the front-end repository [here](https://github.com/beloureiro/AI-CAC-Viewer.git).

The project offers **two main configurations** for running the AI agents:

1. **AI Agent Crew with Ollama Framework**: This option leverages **multiple LLMs** dynamically using the **CrewAI** framework. It selects the most suitable models from a variety of sources (e.g., Meta's LLaMA, NousResearch's Hermes, Microsoft's Phi) to ensure optimal performance for each task. This multi-agent approach allows the system to adapt to the specific needs of different healthcare scenarios by combining the strengths of various models.
   
2. **Python-Only Local LLM Setup**: The second option allows you to run a fully local version of the AI agents using **LM Studio**. This configuration uses local models (such as **Meta-Llama-3.1-8B-Instruct-GGUF**) and processes everything locally in Python. This ensures maximum data security by keeping all operations internal, while also offering granular control over model execution parameters and significant cost savings due to the absence of external API usage.

### Major benefits of this system:
1. **Local LLMs** for **maximum data security**, ensuring that all sensitive healthcare data is processed internally.
2. **Significant cost savings**, as external API calls are not required.
3. **Flexibility to choose the optimal configuration**: The system can switch between using multiple LLMs in combination with **Ollama** and the CrewAI framework or running entirely locally with **Python and LM Studio**.
4. **Granular control with LM Studio**: LM Studio provides precise control over model parameters like temperature, top_p, and repetition penalty, while also supporting GPU optimization for efficient local execution.
5. **Refactored codebase**: Version 1.3 introduces a modular, clean codebase that simplifies further enhancements and maintenance, regardless of the configuration used.

### Benefits of LM Studio Integration

The **LM Studio** integration is especially valuable for those opting for the fully local execution path:

- **Parallel Model Execution**: Run multiple models simultaneously for direct comparison.
- **Fine-tuned Control**: Parameters like temperature, top_p, and repetition penalty can be adjusted to produce more precise outputs.
- **GPU Optimization**: LM Studio efficiently uses NVIDIA/AMD GPUs and supports model quantization to reduce memory use and increase speed.
- **OpenAI-Compatible API**: Allows for seamless integration and real-time streaming of results.
- **Detailed Performance Monitoring**: Track CPU and GPU metrics in real time, allowing for efficient resource management during local execution.

These features make LM Studio highly adaptable for advanced local LLM tasks, ensuring that performance can be compared across models to select the most effective configuration for each use case.

## Agents
Our AI team consists of five dedicated agents:
1. **Patient Experience Expert**: Focuses on understanding patient feedback, identifying key concerns, and measuring emotional intensity to help healthcare providers address critical issues.
2. **Health & IT Process Expert**: Specializes in mapping the patient journey, identifying inefficiencies in workflows, and recommending improvements from both healthcare and IT perspectives.
3. **Clinical Psychologist**: Analyzes the emotional state of patients, developing support strategies that address psychological needs and promote overall well-being.
4. **Communication Expert**: Evaluates the quality of interactions between healthcare professionals and patients, identifying where communication can be improved for better clarity, empathy, and problem resolution.
5. **Manager and Advisor**: Consolidates feedback from all experts, eliminating redundancies and providing clear, actionable reports that offer strategic recommendations for process improvement.

These agents are powered by either a dynamic combination of LLMs, which are tested and swapped across different scenarios using CrewAI, or a fully local execution setup using LM Studio, depending on the chosen configuration.

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation
1. Clone the repository
2. Install the dependencies:

   ```
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