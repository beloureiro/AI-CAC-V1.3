import time  # Adicionando para medição de tempo
from utils import load_yaml_file, load_feedback, send_request, save_results_to_md

class FeedbackAnalyzer:
    def __init__(self, config_path, feedback_path, agents_path):
        # Carregar as configurações do arquivo YAML
        self.config = load_yaml_file(config_path)
        
        # Carregar o feedback do arquivo de texto
        self.feedback = load_feedback(feedback_path)

        # Carregar os agentes do arquivo YAML
        self.agents = load_yaml_file(agents_path)['agents']

        # Configurações da API do modelo
        self.url = self.config['server']['url']
        self.model = self.config['request']['lmstudio_models']['Meta-Llama-3.1-8B-Instruct-GGUF'] #--> escolhe o modelo de acordo com o arquivo de configuração
        self.headers = {"Content-Type": "application/json"}

    def analyze_with_agent(self, agent_prompt, max_tokens, temperature):
        # Substituir o espaço reservado "[feedback]" com o feedback real
        prompt_with_feedback = agent_prompt.replace("[feedback]", self.feedback)
        # Enviar a solicitação ao modelo
        return send_request(self.url, self.headers, self.model, prompt_with_feedback, max_tokens, temperature)

    def run_analysis(self):
        # Dicionário para armazenar os resultados de cada agente
        results = {}

        # Executar cada agente e coletar as respostas
        for agent_name, agent_data in self.agents.items():
            print(f"Running analysis for {agent_name}...")

            # Extrair o prompt, max_tokens e temperatura do YAML
            prompt = agent_data['prompt']
            max_tokens = agent_data.get('max_tokens')
            temperature = agent_data.get('temperature')

            # Verifica se max_tokens e temperature estão definidos para o agente, senão lança um erro
            if max_tokens is None or temperature is None:
                raise ValueError(f"Agente {agent_name} não tem 'max_tokens' ou 'temperature' definidos no YAML.")

            # Fazer a análise com o agente
            result = self.analyze_with_agent(prompt, max_tokens, temperature)
            results[agent_name] = result

            print(f"{agent_name} result: \n{result}\n")
        
        return results

# Caminhos dos arquivos
config_path = "config/local_llm.yaml"
feedback_path = "D:/OneDrive - InMotion - Consulting/AI Projects/AI-CAC-V1.3/patient_feedback.txt"
agents_path = "LmStudio/lm_agents.yaml"

# Medir o tempo de início
start_time = time.time()

# Instanciar o analisador de feedback
analyzer = FeedbackAnalyzer(config_path, feedback_path, agents_path)

# Executar a análise
resultados = analyzer.run_analysis()

# Salvar os resultados no arquivo .md e obter o caminho do arquivo gerado
md_file_path = save_results_to_md(resultados)

# Medir o tempo de término
end_time = time.time()

# Calcular o tempo total
execution_time = end_time - start_time

# Converter para minutos e segundos
minutes, seconds = divmod(execution_time, 60)
execution_message = f"Total execution time: {int(minutes)} minutes and {int(seconds)} seconds."

# Imprimir o tempo total no console
print(execution_message)

# Adicionar o tempo de execução no final do arquivo .md gerado
with open(md_file_path, "a", encoding='utf-8') as file:
    file.write(f"\n{execution_message}\n")
