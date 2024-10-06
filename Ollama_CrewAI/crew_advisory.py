# Instructions:
# If you are using the sequential process (default):
#   - Ensure the manager_agent is included both in the imports and in the agents list.
# If you are using the hierarchical process:
#   - Comment out the manager_agent in the imports and in the agents list, and instead specify it in the manager_agent parameter within the Crew configuration.

from crewai import Crew
from agent_advisory import (
    patient_experience_agent, process_expert_agent, clinical_psychologist_agent, 
    communication_expert_agent, 
    manager_agent  # Manager agent is included by default for sequential process
    # Comment out the line above if using hierarchical process
)
from task_definitions import (
    consolidated_patient_experience_task,  # Consolidated task for Patient Experience Expert
    consolidated_process_task,  # Consolidated task for Process Expert
    consolidated_clinical_psychologist_task,  # Consolidated task for Clinical Psychologist
    consolidated_communication_task,  # Consolidated task for Communication Expert
    consolidated_manager_task  # New consolidated task for Manager and Advisor
)

# Define the embedder as a dictionary, as expected by the Crew class
embedder = {
    "provider": "ollama",
    "config": {
        "model": "nomic-embed-text:latest"  # Match the model you're using with Ollama
    }
}

# Default: Use the sequential process
ai_clinical_crew = Crew(
    agents=[
        patient_experience_agent,
        process_expert_agent,
        clinical_psychologist_agent,
        communication_expert_agent,
        manager_agent  # Manager agent is included as a normal agent in the sequential process
    ],
    tasks=[
        consolidated_patient_experience_task,  # Use consolidated task for Patient Experience Expert
        consolidated_process_task,  # Use consolidated task for Process Expert
        consolidated_clinical_psychologist_task,  # Use consolidated task for Clinical Psychologist
        consolidated_communication_task,  # Use consolidated task for Communication Expert
        consolidated_manager_task  # Use consolidated task for Manager and Advisor
    ],
    process="sequential",  # Default is sequential process to execute tasks one by one
    memory=True,
    embedder=embedder,
    verbose=True
)

# Optional: Use the hierarchical process (uncomment these lines and comment the sequential block to switch to hierarchical)
"""
ai_clinical_crew = Crew(
    agents=[
        patient_experience_agent,
        process_expert_agent,
        clinical_psychologist_agent,
        communication_expert_agent
        # Manager agent is excluded from the agents list in hierarchical mode
    ],
    tasks=[
        consolidated_patient_experience_task,  # Use consolidated task for Patient Experience Expert
        consolidated_process_task,  # Use consolidated task for Process Expert
        consolidated_clinical_psychologist_task,  # Use consolidated task for Clinical Psychologist
        consolidated_communication_task,  # Use consolidated task for Communication Expert
        consolidated_manager_task  # Use consolidated task for Manager and Advisor
    ],
    process="hierarchical",  # Switch to hierarchical to ensure task dependencies are managed
    memory=True,
    embedder=embedder,
    verbose=True,
    respect_context_window=True,  # Ensures the context size is respected
    manager_agent=manager_agent  # Manager agent is explicitly defined here for hierarchical
)
"""

