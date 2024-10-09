import time
from utils import load_yaml_file, load_feedback, send_request, save_results_to_md
from prompt_renderer import render_prompt_with_template

class FeedbackAnalyzer:
    def __init__(self, config_path, feedback_path, agents_path):
        # Load configurations and feedback
        self.config = load_yaml_file(config_path)
        self.feedback = load_feedback(feedback_path)
        self.agents = load_yaml_file(agents_path)['agents']

        # Model API configurations
        self.url = self.config['server']['url']
        self.model = self.config['request']['lmstudio_models']['Meta-Llama-3.1-8B-Instruct-GGUF']
        self.headers = {"Content-Type": "application/json"}

    def analyze_with_agent(self, agent_prompt, max_tokens, temperature, top_k, top_p, repeat_penalty, min_p):
        # Aplicar o template Jinja ao prompt
        final_prompt = render_prompt_with_template(agent_prompt, self.feedback)

        # Enviar a requisição para o modelo
        return send_request(
            self.url, 
            self.headers, 
            self.model, 
            final_prompt, 
            max_tokens, 
            temperature, 
            top_k, 
            top_p, 
            repeat_penalty, 
            min_p
        )

    def run_analysis(self):
        # Dictionary to store results
        results = {}

        # Execute each agent and collect responses
        for agent_name, agent_data in self.agents.items():
            print(f"Running analysis for {agent_name}...")

            # Extract configuration directly from the YAML file
            prompt = agent_data['prompt']
            max_tokens = agent_data['max_tokens']
            temperature = agent_data['temperature']
            top_k = agent_data['top_k']
            top_p = agent_data['top_p']
            repeat_penalty = agent_data['repeat_penalty']
            min_p = agent_data['min_p']

            # Perform the analysis with the agent using parameters from YAML
            result = self.analyze_with_agent(prompt, max_tokens, temperature, top_k, top_p, repeat_penalty, min_p)
            results[agent_name] = result

            print(f"{agent_name} result: \n{result}\n")
        
        return results

# Paths and execution logic
config_path = "config/local_llm.yaml"
feedback_path = "D:/OneDrive - InMotion - Consulting/AI Projects/AI-CAC-V1.3/patient_feedback.txt"
agents_path = "LmStudio/Feedbck_Analyser/lm_agents.yaml"

start_time = time.time()
analyzer = FeedbackAnalyzer(config_path, feedback_path, agents_path)
results = analyzer.run_analysis()
md_file_path = save_results_to_md(results)
end_time = time.time()

execution_time = end_time - start_time
minutes, seconds = divmod(execution_time, 60)
execution_message = f"Total execution time: {int(minutes)} minutes and {int(seconds)} seconds."

print(execution_message)
with open(md_file_path, "a", encoding='utf-8') as file:
    file.write(f"\n{execution_message}\n")
