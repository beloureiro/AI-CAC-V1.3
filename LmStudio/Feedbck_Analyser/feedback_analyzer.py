import time  # Adding for time measurement
from utils import load_yaml_file, load_feedback, send_request, save_results_to_md
from jinja2 import Template  # Importando Jinja2 para usar o template

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

    def apply_jinja_template(self, agent_prompt):
        """Renderiza o template Jinja2 com o prompt do agente."""
        # Template Jinja2 fornecido
        jinja_template_str = """
        {{- bos_token }}
        {%- if custom_tools is defined %}
            {%- set tools = custom_tools %}
        {%- endif %}
        {%- if not tools_in_user_message is defined %}
            {%- set tools_in_user_message = true %}
        {%- endif %}
        {%- if not date_string is defined %}
            {%- set date_string = "26 Jul 2024" %}
        {%- endif %}

        {#- This block extracts the system message, so we can slot it into the right place. #}
        {%- if messages[0]['role'] == 'system' %}
            {%- set system_message = messages[0]['content']|trim %}
            {%- set messages = messages[1:] %}
        {%- else %}
            {%- set system_message = "" %}
        {%- endif %}

        {#- System message + builtin tools #}
        {{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
        {%- if builtin_tools is defined or tools is not none %}
            {{- "Environment: ipython\n" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
        {%- endif %}
        {{- "Cutting Knowledge Date: December 2023\n" }}
        {{- "Today Date: " + date_string + "\n\n" }}
        {%- if tools is not none and not tools_in_user_message %}
            {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
            {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
            {{- "Do not use variables.\n\n" }}
            {%- for t in tools %}
                {{- t | tojson(indent=4) }}
                {{- "\n\n" }}
            {%- endfor %}
        {%- endif %}
        {{- system_message }}
        {{- "<|eot_id|>" }}

        {#- Custom tools are passed in a user message with some extra guidance #}
        {%- if tools_in_user_message and not tools is none %}
            {#- Extract the first user message so we can plug it in here #}
            {%- if messages | length != 0 %}
                {%- set first_user_message = messages[0]['content']|trim %}
                {%- set messages = messages[1:] %}
            {%- else %}
                {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
            {%- endif %}
            {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
            {{- "Given the following functions, please respond with a JSON for a function call " }}
            {{- "with its proper arguments that best answers the given prompt.\n\n" }}
            {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
            {{- "Do not use variables.\n\n" }}
            {%- for t in tools %}
                {{- t | tojson(indent=4) }}
                {{- "\n\n" }}
            {%- endfor %}
            {{- first_user_message + "<|eot_id|>"}}
        {%- endif %}

        {%- for message in messages %}
            {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
                {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
            {%- elif 'tool_calls' in message %}
                {%- if not message.tool_calls|length == 1 %}
                    {{- raise_exception("This model only supports single tool-calls at once!") }}
                {%- endif %}
                {%- set tool_call = message.tool_calls[0].function %}
                {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
                    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
                    {{- "<|python_tag|>" + tool_call.name + ".call(" }}
                    {%- for arg_name, arg_val in tool_call.arguments | items %}
                        {{- arg_name + '="' + arg_val + '"' }}
                        {%- if not loop.last %}
                            {{- ", " }}
                        {%- endif %}
                    {%- endfor %}
                    {{- ")" }}
                {%- else  %}
                    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
                    {{- '{"name": "' + tool_call.name + '", ' }}
                    {{- '"parameters": ' }}
                    {{- tool_call.arguments | tojson }}
                    {{- "}" }}
                {%- endif %}
                {%- if builtin_tools is defined %}
                    {#- This means we're in ipython mode #}
                    {{- "<|eom_id|>" }}
                {%- else %}
                    {{- "<|eot_id|>" }}
                {%- endif %}
            {%- elif message.role == "tool" or message.role == "ipython" %}
                {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
                {%- if message.content is mapping or message.content is iterable %}
                    {{- message.content | tojson }}
                {%- else %}
                    {{- message.content }}
                {%- endif %}
                {{- "<|eot_id|>" }}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
        {%- endif %}
        """
        
        # Substituir a variável feedback diretamente no prompt
        prompt_with_feedback = agent_prompt.replace("[feedback]", self.feedback)

        # Criar o template Jinja2
        template = Template(jinja_template_str)
        
        # Dados fictícios para teste, substitua conforme necessário
        context = {
            "bos_token": "<|begin_of_text|>",
            "builtin_tools": None,
            "messages": [{"role": "system", "content": prompt_with_feedback}],
            "custom_tools": None
        }
        
        # Renderizar o template com o contexto
        rendered_prompt = template.render(**context)
        
        return rendered_prompt

    def analyze_with_agent(self, agent_prompt, max_tokens, temperature, top_k, top_p, repeat_penalty, min_p):
        # Aplicar o template Jinja ao prompt
        final_prompt = self.apply_jinja_template(agent_prompt)

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
        # Dictionary to store the results of each agent
        results = {}

        # Execute each agent and collect the responses
        for agent_name, agent_data in self.agents.items():
            print(f"Running analysis for {agent_name}...")

            # Extract the prompt, max_tokens, temperature, and sampling parameters from the YAML
            prompt = agent_data['prompt']
            max_tokens = agent_data.get('max_tokens')
            temperature = agent_data.get('temperature')
            top_k = agent_data.get('top_k', 40)
            top_p = agent_data.get('top_p', 0.95)
            repeat_penalty = agent_data.get('repeat_penalty', 1.1)
            min_p = agent_data.get('min_p', 0.05)

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
