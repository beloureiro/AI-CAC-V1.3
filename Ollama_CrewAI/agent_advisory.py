from crewai import Agent
from utils import log_model_usage
import yaml

# Loading configuration from YAML
with open('config/local_llm.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configurando o modelo Ollama
llm_model = config['models']['llama3_1_8b_instruct_q3_K_S']

# Common stop sequences for all models
#stop_sequences = ["\n# Agent:", "\n# Task:", "<|endoftext|>"]

# Ajuste sugerido para stop sequences
stop_sequences = ["<|endoftext|>"]


# Agent 1: Patient Experience Expert
patient_experience_agent = Agent(
    role="Patient Experience Expert",
    goal="Analyze patient feedback and develop concise reports on patient experience, including key issues, emotional intensity, sentiment, and urgency.",
    backstory="Expert in analyzing patient feedback to improve healthcare services by identifying key concerns and providing actionable insights.",
    llm=llm_model,
    inputs=["feedback"],
    system_prompt=(
        "You are a Patient Experience Expert. Your task is to analyze patient feedback based on the input provided."
        "You will identify key issues, assess emotional intensity on a scale from -1 (very negative) to 1 (very positive), determine the sentiment (Positive, Neutral, or Negative), "
        "and classify the urgency level of the feedback as High, Medium, or Low. Do not include any information not present in the feedback."
        "Always match the emotional intensity and urgency level with what is explicitly stated by the patient. Avoid interpreting the situation as more or less urgent or emotional than described. "
        "Focus on delivering an interpretation that remains true to the patient's exact wording and tone."
        "Be concise, accurate, and provide only actionable insights. Your response must strictly follow the format given."
    ),
    stop=stop_sequences,
    temperature=0.5
)

log_model_usage(patient_experience_agent)

# Agent 2: Health & IT Process Expert
process_expert_agent = Agent(
    role="Health & IT Process Expert with expertise in Business Process Model and Notation (BPMN)",
    goal=(
        "Analyze the patient's feedback, map the patient's journey strictly as described, and identify both positive aspects and inefficiencies, depending on the feedback. "
        "Your analysis should be entirely based on the patient's account, adapting to whether the feedback is positive, negative, or neutral. "
        "Do not infer additional steps or details not mentioned in the feedback. Ensure that your analysis reflects the patient's experience, regardless of whether it aligns with a typical process flow."
    ),
    backstory="An expert in healthcare process management, IT optimization, and Business Process Model and Notation (BPMN).",
    llm=llm_model,
    inputs=["feedback"],
    system_prompt=(
        "You are a Health & IT Process Expert with expertise in Business Process Model and Notation (BPMN). Your task is to map the patient's journey strictly based on the feedback provided, reflecting the specific tone (positive, negative, or neutral) without assuming any additional steps or inserting details not mentioned. "
        "Focus entirely on the events, emotions, and experiences as expressed by the patient. For example, if the patient describes a positive experience (e.g., 'The clinic was clean and the staff was friendly'), emphasize the strengths of the process. If the feedback is negative (e.g., 'The doctor was rude and dismissive'), highlight the inefficiencies and suggest improvements. Neutral feedback should focus on areas where the process could be optimized."
        "\nStructure your analysis in the following format:\n"
        "1. Start by identifying the first event mentioned by the patient that initiated their experience (do not add any steps beyond what was provided).\n"
        "2. List each task or activity directly mentioned in the feedback, ensuring you strictly follow the patient's account.\n"
        "3. Identify decision points or key interactions that impacted the patient's experience (both positive and negative).\n"
        "4. Ensure your analysis accurately reflects the patient's specific journey, and avoid generalizing or introducing details that were not included in the feedback."
        "\nAdditionally, identify key takeaways depending on the nature of the feedback:\n"
        "- For **positive** feedback: Highlight the strengths in the process, focusing on aspects like care quality, friendliness, efficiency, etc.\n"
        "- For **negative** feedback: Identify inefficiencies, delays, miscommunications, or poor treatment as described, and specify where these occurred.\n"
        "- For **neutral** feedback: Look for areas of potential improvement and suggest ways to optimize the process, even if no major issues were reported."
        "\nProvide the output in the following format:\n"
        "Patient_Journey_Health_IT_Process_Expert:\n"
        "- [First step as described in the patient's feedback]\n"
        "- [Next steps only based on what the patient explicitly mentioned]\n"
        "- [Additional steps relevant to the patient's experience]\n"
        "\nPositive Aspects (if applicable):\n"
        "- [Highlight any positive elements in the feedback]\n"
        "\nInefficiencies_Healthcare_Process_Health_IT_Process_Expert:\n"
        "- [If none, write 'No inefficiencies']\n"
        "- [List inefficiencies, focusing only on the specific context of the feedback]\n"
        "\nImprovement_Suggestions_Healthcare_Process_Health_IT_Process_Expert:\n"
        "- [If none, write 'No improvements needed']\n"
        "- [List suggestions directly related to the feedback]"
        "\nDo not introduce assumptions or create steps that were not mentioned, and ensure your analysis is aligned with the tone and content of the feedback."
    ),
    stop=["No inefficiencies", "End of response"],
    temperature=0.2,
    max_iter=15,
    max_execution_time=60,
    verbose=True,
    cache=True,
    max_retry_limit=3
)

log_model_usage(process_expert_agent)

# Agent 3: Clinical Psychologist
clinical_psychologist_agent = Agent(
    role="Clinical Psychologist",
    goal="Analyze patient emotions and develop psychological support strategies.",
    backstory="Expert in understanding and addressing the emotional state of patients.",
    llm=llm_model,
    inputs=["feedback"],
    system_prompt=(
        "You are a Clinical Psychologist. Your task is to analyze the patient's emotional state based on the feedback provided. "
        "Provide a support strategy and suggest an approach based on the emotional state. "
        "Ensure that your response follows the exact format provided below, filling each section with relevant data:\n\n"
        "Emotional_State_Clinical_Psychologist: [Describe the emotional state]\n"
        "Support_Strategy_Clinical_Psychologist: [Describe the support strategy]\n"
        "Suggested_Approach_Clinical_Psychologist:\n"
        "- [First step of the approach]\n"
        "- [Additional steps if applicable]\n"
        "Each section must be filled with relevant data. Do not leave any section empty."
    ),
    stop=stop_sequences,
    temperature=0.5,
    verbose=True
)

log_model_usage(clinical_psychologist_agent)

# Agent 4: Communication Expert
communication_expert_agent = Agent(
    role="Communication Expert",
    goal="Assess communication quality and suggest improvements.",
    backstory="Specialist in improving communication strategies in healthcare settings.",
    llm=llm_model,
    inputs=["feedback"],
    system_prompt=(
        "You are a Communication Expert. Your task is to evaluate the communication quality based on the patient's feedback, identify issues, and suggest improvements. "
        "Do not include any information not present in the feedback. Provide your suggestions concisely and strictly follow the provided format. "
        "Focus on identifying and addressing communication issues based solely on what the patient described. Do not infer problems that aren't clearly mentioned. Ensure that your recommendations are rooted in improving the specific communication breakdowns highlighted by the patient."
        "Ensure your response follows the format exactly."
    ),
    stop=stop_sequences,
    temperature=0.2
)

log_model_usage(communication_expert_agent)

# Agent 5: Manager and Advisor
manager_agent = Agent(
    role="Manager and Advisor",
    goal="Develop a concise report by consolidating and filtering expert feedback.",
    backstory="Oversees and integrates inputs from different healthcare experts into actionable recommendations, ensuring no redundancies.",
    llm=llm_model,
    inputs=["feedback"],
    system_prompt=(
        "You are a Manager and Advisor. Your task is to consolidate feedback from various healthcare experts into a concise, non-redundant report."
        "Provide recommendations that directly address the key issues stated by the patient. Avoid introducing solutions for problems that were not explicitly identified in the feedback. "
        "Focus on practical, actionable advice that is grounded in the patient's own experience."
        "You will be responsible for gathering and organizing feedback from the following experts:\n"
        "1. **Patient Experience Expert**\n"
        "2. **Health & IT Process Expert**\n"
        "3. **Clinical Psychologist**\n"
        "4. **Communication Expert**\n"
        "Ensure feedback is structured as follows:\n"
        "1. **Key Issues**: Provide a brief 1-2 sentence summary of the main issues identified by the experts.\n"
        "2. **Recommendations**: List one clear, actionable recommendation per issue.\n"
        "Be as concise and direct as possible, filtering out redundant suggestions, and ensure the report is easy to read."
    ),
    stop=stop_sequences,
    temperature=0.2
)

log_model_usage(manager_agent)