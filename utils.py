import os
import json
from datetime import datetime


def get_patient_feedback():
    with open("patient_feedback.txt", "r", encoding='utf-8') as file:
        return file.read().strip()


def log_all_models(agents):
    for agent in agents:
        if isinstance(agent, str):
            print(f"Agent: {agent}, Model: Unknown Model")
        elif hasattr(agent, 'role') and hasattr(agent, 'llm'):
            model_name = agent.llm if agent.llm else "Unknown Model"
            print(f"Agent: {agent.role}, Model: {model_name}")
        else:
            print("Unknown agent type or missing attributes.")


def format_task_descriptions(tasks, feedback):
    for task in tasks:
        if '{feedback}' in task.description:
            task.description = task.description.format(feedback=feedback)


def execute_agents(tasks_output):
    for task_result in tasks_output:
        agent_name = task_result.agent if isinstance(task_result.agent, str) else task_result.agent.role
        print(f"############################")
        print(f"# Agent: {agent_name}")
        response = task_result.raw if hasattr(task_result, 'raw') else "No response available"
        print(f"## Final Answer:\n{response}\n")


def save_consolidated_report(patient_feedback, tasks_output, total_duration):
    directory = r"D:\OneDrive - InMotion - Consulting\AI Projects\AI-CAC-V1.2\data_reports_txt"
    file_name = f"{directory}\\report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(file_name, "w", encoding='utf-8') as report_file:
        report_file.write("############################\n")
        report_file.write("# AI Clinical Advisory Crew Report\n")
        report_file.write("############################\n\n")
        report_file.write(f"Patient Feedback: {patient_feedback}\n\n")
        for task_result in tasks_output:
            agent_name = task_result.agent if isinstance(task_result.agent, str) else task_result.agent.role
            response = task_result.raw if hasattr(task_result, 'raw') else "No response available"
            report_file.write(f"############################\n")
            report_file.write(f"# Agent: {agent_name}\n")
            report_file.write(f"## Final Answer:\n{response}\n\n")
        minutes, seconds = divmod(int(total_duration), 60)
        report_file.write(f"Total execution time: {minutes} minutes and {seconds} seconds.\n")
        report_file.write("############################\n")
        report_file.write("# Consolidated Final Report\n")
        report_file.write("############################\n")
        report_file.write("\nDisclaimer\n")
        report_file.write("The analyses in this report were conducted by different LLM models in training mode, which take patient feedback as absolute truth. ")
        report_file.write("Feedback reflects the patient's individual perception and, in some cases, may not capture the full complexity of the situation, including institutional and contextual factors.\n")
        report_file.write("AI Clinical Advisory Crew framework, beyond providing technical analyses, acts as a strategic driver, steering managerial decisions across various operational areas.\n")
        report_file.write("The reader is responsible for validating the feasibility of the suggested actions and their alignment with stakeholder objectives.\n")
    return file_name


def save_agent_results_as_json(patient_feedback, tasks_output, total_duration):
    directory = r"D:\OneDrive - InMotion - Consulting\AI Projects\AI-CAC-V1.2\data_reports_json"
    file_name = f"{directory}\\report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_data = {
        "patient_feedback": patient_feedback,
        "total_execution_time": f"{int(total_duration // 60)} minutes and {int(total_duration % 60)} seconds",
        "agents": []
    }
    for task_result in tasks_output:
        agent_name = task_result.agent if isinstance(task_result.agent, str) else task_result.agent.role
        response = task_result.raw if hasattr(task_result, 'raw') else "No response available"
        agent_data = process_agent_response(response, agent_name)
        report_data["agents"].append(agent_data)
    with open(file_name, "w", encoding='utf-8') as json_file:
        json.dump(report_data, json_file, ensure_ascii=False, indent=4)
    return file_name


def process_agent_response(response, agent_name):
    agent_data = {
        "agent_name": agent_name,
        "response": {}
    }
    if response and isinstance(response, str):
        lines = response.split('\n')
        current_field = None
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = map(str.strip, line.split(':', 1))
                current_field = key
                agent_data["response"][current_field] = value
            elif line.startswith('-') and current_field:
                if isinstance(agent_data["response"][current_field], str):
                    agent_data["response"][current_field] = [agent_data["response"][current_field]]
                agent_data["response"][current_field].append(line.strip('- '))
            elif current_field:
                if isinstance(agent_data["response"][current_field], list):
                    agent_data["response"][current_field][-1] += f" {line}"
                else:
                    agent_data["response"][current_field] += f" {line}"
    return agent_data


def log_model_usage(agents):
    # Implementação da função para registrar o uso dos modelos
    for agent in agents:
        print(f"Model used by agent: {agent}")