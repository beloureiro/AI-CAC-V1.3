---
### What's New in Version 1.3
#### Introducing RAGBot: A Healthcare Coaching Assistant

Version 1.3 introduces **RAGBot**, a **Retrieval-Augmented Generation (RAG)** chatbot. This new feature combines **SentenceTransformer (all-MiniLM-L6-v2)** for generating text embeddings with a **FAISS** index to quickly retrieve relevant expert feedback. RAGBot acts as a personalized coaching assistant for healthcare professionals.

#### Key Technical Features:
- **New Chat Functionality**: Users can now interact with RAGBot in real-time, receiving expert-driven, actionable feedback on performance.
- **Efficient Data Handling**: Markdown reports are consolidated, split into chunks, and indexed for fast, context-driven responses.
- **Contextual Response Generation**: RAGBot retrieves relevant text chunks using FAISS and generates personalized responses using advanced models to maintain dynamic interactions.
- **Modular Design**: The architecture separates data processing, embedding generation, and FAISS indexing, enhancing maintainability.
- **API-based LLM Integration**: Custom API interactions with configurable parameters (e.g., temperature, token limits) ensure precise responses tailored to user queries.

This version significantly expands functionality, introducing real-time, context-aware interactions to assist healthcare professionals.

---
# AI Clinical Advisory Crew (v1.3)

## Description
Welcome to the **AI Clinical Advisory Crew**, an advanced and flexible system designed to transform and elevate the patient experience in healthcare. This project brings together a team of specialized AI agents, each with a unique role in analyzing patient feedback, improving healthcare workflows, assessing emotional states, and delivering actionable recommendations for communication and operational improvements.

For a live demonstration of how this system works in practice, visit the frontend Viewer platform that visualizes AI agent outputs: [AI-CAC Viewer](https://ai-cac.streamlit.app/). This will give you a practical understanding of the AI analysis in action and how the system improves healthcare outcomes. You can also explore the front-end repository [here](https://github.com/beloureiro/AI-CAC-Viewer.git).

At the heart of this system lies its dynamic flexibility: it navigates across a suite of **Large Language Models (LLMs)** to determine the optimal AI configuration for each specific task. By utilizing models like Meta's LLaMA, NousResearch's Hermes, Microsoft's Phi, and others, the system continuously tests and refines outputs to ensure that the best-suited AI crew is selected to address the task at hand. This multi-agent approach allows for the combination of strengths across different models, ensuring comprehensive, data-driven analysis that adapts to the unique needs of your healthcare environment.

**Major benefits** of this system:
1. **Local LLMs** for **maximum data security** by processing all information internally, without reliance on third-party APIs.
2. **Significant cost savings**, as there is no need for external API usage.
3. **Refactored codebase**: Version 1.3 includes a new complete refactoring of the code, making it more modular, clean, and organized, which simplifies further enhancements and maintenance.
4. **Python resource with LM Studio**:
   - I’ve integrated LM Studio, offering a robust platform for testing new models and comparing their performance.
   - This allows for direct comparisons between models and improved configurability for advanced local LLM execution.

### Benefits of LM Studio Integration

**LM Studio** offers key benefits for local LLM execution:

- **Parallel Model Execution**: Run multiple models simultaneously for direct comparison.
- **Granular Control**: Fine-tune parameters like temperature, top_p, and repetition penalty for precise output.
- **GPU Optimization**: Efficiently uses NVIDIA/AMD GPUs and supports model quantization to reduce memory use.
- **API Compatibility**: Works with OpenAI-like API for seamless integration and real-time streaming.
- **Performance Monitoring**: Provides detailed CPU/GPU metrics during execution.

These features make LM Studio highly adaptable for advanced LLM tasks, ensuring we can compare performance metrics and choose the most suitable AI models efficiently.

## Agents
Our AI team consists of five dedicated agents:
1. **Patient Experience Expert**: Focuses on understanding patient feedback, identifying key concerns, and measuring emotional intensity, ensuring that healthcare providers can address critical issues.
2. **Health & IT Process Expert**: Specializes in mapping the patient journey, identifying inefficiencies in workflows, and recommending improvements from both the healthcare and IT perspectives.
3. **Clinical Psychologist**: Analyzes the emotional state of patients, developing support strategies that address psychological needs and promote overall well-being.
4. **Communication Expert**: Evaluates the quality of interactions between healthcare professionals and patients, identifying where communication can be improved for better clarity, empathy, and problem resolution.
5. **Manager and Advisor**: Consolidates feedback from all experts, eliminating redundancies and providing clear, actionable reports that offer strategic recommendations for process improvement.

These agents are powered by a diverse range of LLMs, which are dynamically tested and swapped across different scenarios to identify the most suitable technology for each specific task.

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