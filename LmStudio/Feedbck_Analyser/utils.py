import requests
import json
import yaml
import os
from datetime import datetime

def load_yaml_file(filepath):
    """Load YAML files."""
    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_feedback(filepath):
    """Load feedback from a text file."""
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

def send_request(url, headers, model, prompt, max_tokens, temperature):
    """Send a POST request to the model."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Check the response status
    if response.status_code == 200:
        completion = response.json()
        return completion['choices'][0]['text']
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def save_results_to_md(results):
    """Save the analysis results to a Markdown file."""
    # Get the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the .md file path based on the specified folder and the format "report_date_time.md"
    md_file_path = os.path.join(
        "D:/OneDrive - InMotion - Consulting/AI Projects/AI-CAC-V1.3/LmStudio/lms_reports_md", 
        f'report_{current_time}.md'
    )

    # Open the file in write mode (create or overwrite)
    with open(md_file_path, 'w', encoding='utf-8') as file:
        file.write("# Feedback Analysis Report\n\n")
        file.write("This report contains the analysis results for each expert agent based on the patient's feedback.\n\n")
        
        for agent, result in results.items():
            file.write(f"## {agent}\n")
            file.write(f"{result}\n")

    print(f"Results saved to {md_file_path}")
    
    # Return the file path for later use
    return md_file_path
