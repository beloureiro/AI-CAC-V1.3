import time  # Adding for time measurement
from utils import load_yaml_file, load_feedback, send_request, save_results_to_md

class FeedbackAnalyzer:
    def __init__(self, config_path, feedback_path, agents_path):
        # Load the configuration from the YAML file
        self.config = load_yaml_file(config_path)
        
        # Load the feedback from the text file
        self.feedback = load_feedback(feedback_path)

        # Load the agents from the YAML file
        self.agents = load_yaml_file(agents_path)['agents']

        # Model API configurations
        self.url = self.config['server']['url']
        self.model = self.config['request']['lmstudio_models']['Meta-Llama-3.1-8B-Instruct-GGUF']  # Choose the model according to the configuration file
        self.headers = {"Content-Type": "application/json"}

    def analyze_with_agent(self, agent_prompt, max_tokens, temperature, top_k, top_p, repeat_penalty, min_p):
        # Replace the placeholder "[feedback]" with the actual feedback
        prompt_with_feedback = agent_prompt.replace("[feedback]", self.feedback)
        # Send the request to the model
        return send_request(
            self.url, 
            self.headers, 
            self.model, 
            prompt_with_feedback, 
            max_tokens, 
            temperature, 
            top_k, 
            top_p, 
            repeat_penalty, 
            min_p
        )

    def run_analysis(self):
        # Dictionary to store the results of each agent
        results = {}

        # Execute each agent and collect the responses
        for agent_name, agent_data in self.agents.items():
            print(f"Running analysis for {agent_name}...")

            # Extract the prompt, max_tokens, temperature, and sampling parameters from the YAML
            prompt = agent_data['prompt']
            max_tokens = agent_data.get('max_tokens')
            temperature = agent_data.get('temperature')
            top_k = agent_data.get('top_k', 40)  # Default value if not provided in the YAML
            top_p = agent_data.get('top_p', 0.95)
            repeat_penalty = agent_data.get('repeat_penalty', 1.1)
            min_p = agent_data.get('min_p', 0.05)

            # Check if max_tokens and temperature are defined for the agent, otherwise raise an error
            if max_tokens is None or temperature is None:
                raise ValueError(f"Agent {agent_name} does not have 'max_tokens' or 'temperature' defined in the YAML.")

            # Perform the analysis with the agent, including sampling parameters
            result = self.analyze_with_agent(prompt, max_tokens, temperature, top_k, top_p, repeat_penalty, min_p)
            results[agent_name] = result

            print(f"{agent_name} result: \n{result}\n")
        
        return results

# File paths
config_path = "config/local_llm.yaml"
feedback_path = "D:/OneDrive - InMotion - Consulting/AI Projects/AI-CAC-V1.3/patient_feedback.txt"
agents_path = "LmStudio/Feedbck_Analyser/lm_agents.yaml"

# Measure the start time
start_time = time.time()

# Instantiate the feedback analyzer
analyzer = FeedbackAnalyzer(config_path, feedback_path, agents_path)

# Run the analysis
results = analyzer.run_analysis()

# Save the results to the .md file and get the path of the generated file
md_file_path = save_results_to_md(results)

# Measure the end time
end_time = time.time()

# Calculate the total time
execution_time = end_time - start_time

# Convert to minutes and seconds
minutes, seconds = divmod(execution_time, 60)
execution_message = f"Total execution time: {int(minutes)} minutes and {int(seconds)} seconds."

# Print the total time to the console
print(execution_message)

# Append the execution time at the end of the generated .md file
with open(md_file_path, "a", encoding='utf-8') as file:
    file.write(f"\n{execution_message}\n")
