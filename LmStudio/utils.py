import requests
import json
import yaml
import os
from datetime import datetime

def load_yaml_file(filepath):
    """Carregar arquivos YAML."""
    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_feedback(filepath):
    """Carregar feedback de um arquivo de texto."""
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

def send_request(url, headers, model, prompt, max_tokens, temperature):
    """Enviar a requisição POST para o modelo."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    # Fazer a requisição POST
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Verificar o status da resposta
    if response.status_code == 200:
        completion = response.json()
        return completion['choices'][0]['text']
    else:
        print(f"Erro: {response.status_code}")
        print(response.text)
        return None

def save_results_to_md(results):
    """Salvar os resultados da análise em um arquivo Markdown."""
    # Pega a data e hora atuais
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define o caminho do arquivo .md com base na pasta especificada e no formato "report_data_hora.md"
    md_file_path = os.path.join(
        "D:/OneDrive - InMotion - Consulting/AI Projects/AI-CAC-V1.2/LmStudio/lms_reports_md", 
        f'report_{current_time}.md'
    )

    # Abrir o arquivo em modo de escrita (criar ou sobrescrever)
    with open(md_file_path, 'w', encoding='utf-8') as file:
        file.write("# Feedback Analysis Report\n\n")
        file.write("This report contains the analysis results for each expert agent based on the patient's feedback.\n\n")
        
        for agent, result in results.items():
            file.write(f"## {agent}\n\n")
            file.write(f"```\n{result}\n```\n\n")

    print(f"Results saved to {md_file_path}")
    
    # Retorna o caminho do arquivo para uso posterior
    return md_file_path
