from crewai import Task
from agent_advisory import (
    patient_experience_agent, process_expert_agent, clinical_psychologist_agent,
    communication_expert_agent, manager_agent
)

# Agent 1 Tasks: Patient Experience Expert
consolidated_patient_experience_task = Task(
    description=(
        "Always match the emotional intensity and urgency level with what is explicitly stated by the patient. Avoid interpreting the situation as more or less urgent or emotional than described. Focus on delivering an interpretation that remains true to the patient's exact wording and tone."
        "As a Patient Experience Expert, analyze the following patient feedback and provide the information in the specified format. "
        "Ensure that you insert the data directly into the corresponding keys/columns without adding any explanations, commentary, or unnecessary details outside of the expected format:\n\n"
        "'{feedback}'\n\n"
        "Depending on the sentiment of the feedback (positive, negative, or neutral), adjust your analysis accordingly:\n"
        "- If the feedback is **positive**, focus on identifying key positive aspects of the patient's experience.\n"
        "- If the feedback is **negative**, identify areas where the experience can be improved.\n"
        "- If the feedback is **neutral**, look for potential areas of improvement or consistency in the patient’s experience.\n\n"
        "**Insert the information directly into the format below. Do not write additional comments or phrases outside of the keys/columns. Only the data should be provided in the exact format below.**\n\n"
        "Sentiment_Patient_Experience_Expert: [Positive/Neutral/Negative]\n"
        "Emotional_Intensity_Patient_Experience_Expert: [Score from -1 to 1, where -1 is highly negative and 1 is highly positive]\n"
        "Urgency_Level_Patient_Experience_Expert: [High/Medium/Low]\n"
        "Key_Issues_Patient_Experience_Expert:\n"
        "- [First key issue directly from feedback]\n"
        "- [Second key issue if applicable]"
    ),
    expected_output=(
        "Sentiment_Patient_Experience_Expert: [Positive/Neutral/Negative]\n"
        "Emotional_Intensity_Patient_Experience_Expert: [Numeric score from -1 to 1]\n"
        "Urgency_Level_Patient_Experience_Expert: [High/Medium/Low]\n"
        "Key_Issues_Patient_Experience_Expert:\n"
        "- [Key issue]\n"
        "- [Another key issue if applicable]"
    ),
    agent=patient_experience_agent,
    inputs={"feedback"},
    force_output=True,
    output_format="text",
    max_tokens=150
)

# Agent 2 Tasks: Health & IT Process Expert
consolidated_process_task = Task(
    description=(
        "Strictly follow the BPMN methodology when mapping the patient’s journey as described in the feedback. "
        "You must adapt your analysis based on whether the feedback is positive, negative, or neutral, without introducing any additional steps not explicitly mentioned by the patient.\n"
        "As a Health & IT Process Expert, your task is to map the patient’s journey by focusing on the details of their feedback. If the patient mentions positive aspects (e.g., friendliness of staff), highlight these strengths. If the feedback is negative (e.g., long wait times, poor communication), identify the inefficiencies and suggest relevant improvements. "
        "For neutral feedback, focus on areas that could be improved to optimize the process, even if no specific issues were raised.\n\n"
        "Adjust your analysis based on the tone and content of the feedback:\n"
        "- For **positive** feedback, focus on the aspects of the process that were successful and contributed to a positive experience.\n"
        "- For **negative** feedback, identify where the breakdown in the process occurred, specifying the exact steps and transitions that led to the issue.\n"
        "- For **neutral** feedback, examine the patient’s journey for areas of optimization and suggest potential improvements.\n\n"
        "'{feedback}'\n\n"
        "**Ensure strict adherence to BPMN principles while focusing on the patient's feedback. Prioritize the patient’s actual experience and avoid generalizing or assuming steps not mentioned in the feedback.**\n\n"
        "Patient_Journey_Health_IT_Process_Expert:\n"
        "- [First step based on the patient’s experience]\n"
        "- [Second step, reflecting transitions relevant to the feedback]\n"
        "- [Additional steps mentioned in the feedback]\n"
        "Positive Aspects (if applicable):\n"
        "- [Highlight any positive feedback provided]\n"
        "Inefficiencies_Healthcare_Process_Health_IT_Process_Expert:\n"
        "- [If none, write 'No inefficiencies']\n"
        "- [Identify inefficiencies based on the negative aspects of the feedback]\n"
        "Improvement_Suggestions_Healthcare_Process_Health_IT_Process_Expert:\n"
        "- [If none, write 'No improvements needed']\n"
        "- [List suggestions relevant to the feedback]"
    ),
    expected_output=(
        "Patient_Journey_Health_IT_Process_Expert:\n"
        "- [First step directly linked to feedback]\n"
        "- [Second step based on feedback transitions]\n"
        "- [Additional steps as applicable based on feedback]\n"
        "Positive Aspects (if applicable):\n"
        "- [Highlight positive elements]\n"
        "Inefficiencies_Healthcare_Process_Health_IT_Process_Expert:\n"
        "- No inefficiencies\n"
        "- [List inefficiencies if applicable, based on feedback]\n"
        "Improvement_Suggestions_Healthcare_Process_Health_IT_Process_Expert:\n"
        "- No improvements needed\n"
        "- [List improvements based on feedback]"
    ),
    agent=process_expert_agent,
    inputs={"feedback"},
    force_output=True,
    output_format="text",
    max_tokens=300
)


# Agent 3 Tasks: Clinical Psychologist
consolidated_clinical_psychologist_task = Task(
    description=(
        "Analyze the patient's emotional state based on the feedback provided and develop a tailored psychological support strategy. "
        "Propose an approach to address the emotional impact as described by the patient, ensuring no assumptions are made beyond the expressed emotions.\n\n"
        "Depending on the sentiment of the feedback (positive, negative, or neutral), adjust your support strategy accordingly:\n"
        "- For **positive** feedback, focus on reinforcing emotional well-being.\n"
        "- For **negative** feedback, focus on addressing emotional concerns and improving the patient’s well-being.\n"
        "- For **neutral** feedback, suggest strategies to maintain emotional balance or address subtle issues.\n\n"
        "**Strictly follow the format below without adding extra comments or text.**\n\n"
        "Emotional_State_Clinical_Psychologist: [Emotional state]\n"
        "Support_Strategy_Clinical_Psychologist: [Support strategy]\n"
        "Suggested_Approach_Clinical_Psychologist:\n"
        "- [Approach step]\n"
        "- [Additional approach steps if applicable]"
    ),
    expected_output=(
        "Emotional_State_Clinical_Psychologist: [Emotional state]\n"
        "Support_Strategy_Clinical_Psychologist: [Support strategy]\n"
        "Suggested_Approach_Clinical_Psychologist:\n"
        "- [Approach step]\n"
        "- [Another approach step if applicable]"
    ),
    agent=clinical_psychologist_agent,
    inputs={"feedback"},
    force_output=True,
    output_format="text",
    max_tokens=250
)


# Agent 4 Tasks: Communication Expert
consolidated_communication_task = Task(
    description=(
        "Focus on identifying and addressing communication issues based solely on what the patient described. Do not infer problems that aren't clearly mentioned. Ensure that your recommendations are rooted in improving the specific communication breakdowns highlighted by the patient."
        "As a Communication Expert, evaluate the communication quality based on the patient feedback, identify any issues, and suggest improvements. Provide a final recommendation for improving communication strategies:\n\n"
        "'{feedback}'\n\n"
        "Depending on the sentiment of the feedback (positive, negative, or neutral), adjust your communication analysis and suggestions accordingly:\n"
        "- If the feedback is **positive**, focus on identifying best communication practices and reinforcing them.\n"
        "- If the feedback is **negative**, focus on addressing communication breakdowns and propose improvements.\n"
        "- If the feedback is **neutral**, suggest strategies to maintain or slightly improve the quality of communication.\n\n"
        "**Ensure that the data is inserted directly into the format specified below, without adding any additional commentary or explanation.**\n\n"
        "Communication_Quality_Communication_Expert: [Excellent/Good/Fair/Poor]\n"
        "Issues_Identified_Communication_Expert:\n"
        "- [First issue identified]\n"
        "- [Second issue if applicable]\n"
        "Suggested_Improvements_Communication_Expert:\n"
        "- [First improvement suggestion]\n"
        "- [Second suggestion if applicable]\n"
        "Final_Recommendation_Communication_Expert: [Overall recommendation]"
    ),
    expected_output=(
        "Communication_Quality_Communication_Expert: [Quality]\n"
        "Issues_Identified_Communication_Expert:\n"
        "- [Issue]\n"
        "- [Another issue if applicable]\n"
        "Suggested_Improvements_Communication_Expert:\n"
        "- [Improvement]\n"
        "- [Another improvement if applicable]\n"
        "Final_Recommendation_Communication_Expert: [Recommendation]"
    ),
    agent=communication_expert_agent,
    inputs={"feedback"},
    force_output=True,
    output_format="text",
    max_tokens=150
)

# Agent 5 Tasks: Manager and Advisor
consolidated_manager_task = Task(
    description=(
        "Provide recommendations that directly address the key issues stated by the patient. Avoid introducing solutions for problems that were not explicitly identified in the feedback. Focus on practical, actionable advice that is grounded in the patient's own experience."

        "As a Manager and Advisor, create a concise report by filtering out redundant recommendations from the expert feedback. Summarize the key points in bullet points, with no more than two sentences per item:\n\n"
        "'{feedback}'\n\n"
        "Depending on the sentiment of the feedback (positive, negative, or neutral), adjust your summary accordingly:\n"
        "- If the feedback is **positive**, focus on key successes and suggestions to maintain them.\n"
        "- If the feedback is **negative**, highlight key issues and propose effective solutions.\n"
        "- If the feedback is **neutral**, summarize the feedback with an emphasis on maintaining or improving operational consistency.\n\n"
        "**Insert the data directly into the format provided, without additional comments or narrative. Ensure that each entry corresponds directly to the key issue or recommendation identified.**\n\n"
        "Key_Issues_Manager_and_Advisor:\n"
        "- [First key issue from experts]\n"
        "- [Second key issue if applicable]\n"
        "Recommendations_Manager_and_Advisor:\n"
        "- [First recommendation corresponding to key issue]\n"
        "- [Second recommendation if applicable]"
    ),
    expected_output=(
        "Key_Issues_Manager_and_Advisor:\n"
        "- [Issue]\n"
        "- [Another issue if applicable]\n"
        "Recommendations_Manager_and_Advisor:\n"
        "- [Recommendation]\n"
        "- [Another recommendation if applicable]"
    ),
    agent=manager_agent,
    inputs={"feedback"},
    force_output=True,
    output_format="text",
    max_tokens=150
)
