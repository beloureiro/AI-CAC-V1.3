import time
from crew_advisory import ai_clinical_crew
from utils import (
    get_patient_feedback, log_all_models, format_task_descriptions, 
    execute_agents, save_consolidated_report, save_agent_results_as_json
)

def execute_crew():
    start_time = time.time()
    patient_feedback = get_patient_feedback()

    print("############################")
    print("# AI Clinical Advisory Crew Report")
    print("############################\n")
    print(f"Patient Feedback for Analysis:\n{patient_feedback}\n")

    format_task_descriptions(ai_clinical_crew.tasks, patient_feedback)

    print("LLM Models Used by Agents:\n")
    agents = [task.agent for task in ai_clinical_crew.tasks]
    log_all_models(agents)

    result = ai_clinical_crew.kickoff(inputs={"feedback": patient_feedback})
    execute_agents(result.tasks_output)

    total_duration = time.time() - start_time

    print("############################")
    print("# Consolidated Final Report")
    print("############################\n")
    print(f"Patient Feedback: {patient_feedback}\n")

    txt_file_name = save_consolidated_report(patient_feedback, result.tasks_output, total_duration)
    print(f"Report saved as: {txt_file_name}")

    json_file_name = save_agent_results_as_json(patient_feedback, result.tasks_output, total_duration)
    print(f"Report saved as JSON: {json_file_name}")

if __name__ == "__main__":
    execute_crew()
