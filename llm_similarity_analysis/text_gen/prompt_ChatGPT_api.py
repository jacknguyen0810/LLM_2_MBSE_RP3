import os
from openai import OpenAI


class PromptChatGPT2Text:
    """
    This class is to be used to prompt the chatGPT API to produce either the system modes,
    functions, subsystems (if system requirements are given), or components (if subsystem
    requirements are given
    """
    def __init__(self, input_fp: str, prompt: str = None, output_fp: str = None, filename: str = "llm_output.txt", runs: int = 1, model: str = 'gpt-3.5-turbo-0125', output_start_number = 129) -> None:
        # Self variables
        self.client = OpenAI()
        self.runs = runs
        self.model = model
        self.input_tokens = None
        self.output_tokens = None
        self.filename = filename
        self.output_counter = output_start_number
        
        # Raise errors for incorrect values. 
        if output_fp is None:
            self.output_fp = os.getcwd()
        else:
            self.output_fp = output_fp
        if prompt is None:
            raise ValueError("No prompt has been inputted.")
        else:
            self.prompt = prompt
        
        # Open the .txt file containing the requirements and turn it into a string
        # Check if input_fp is a .txt file
        if not input_fp.endswith('.txt'):
            raise ValueError("Input file is not a .txt file.")
        with open(input_fp, encoding='utf8') as f:
            self.requirements = f.read()
            
        # Call API and generate responses.
        self.run()
        
        model_pricing = {
            "gpt-4-0125-preview": [0.01, 0.03],
            "gpt-4-1106-preview": [0.01, 0.03],
            "gpt-4-1106-vision-preview": [0.01, 0.03],
            "gpt-4": [0.03, 0.06],
            "gpt-4-32k": [0.06, 0.12],
            "gpt-3.5-turbo-0125": [0.0005, 0.0015],
            "gpt-3.5-turbo-instruct": [0.0015, 0.0020]
        }
        
        # Calculate Pricing
        input_cost = self.input_tokens // 1000 * model_pricing[model][0]
        output_cost = self.output_tokens // 1000 * model_pricing[model][1]
        
        print("\nUsage Info:\n")
        print(f"\nInput Tokens: {self.input_tokens} \nOutput Tokens: {self.output_tokens}")
        print(f"\nTotal Tokens: {self.output_tokens+ self.input_tokens}")
        print("\n\nApproximate Cost\n")
        print(f"\nInput Cost: ${input_cost} \nOutput Cost: ${output_cost}")
        print(f"\nTotal Cost: ${input_cost + output_cost}")
        
        
        
    def run(self) -> None:         
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a systems engineer, skilled in designing a complex system to meet a set of requirements."},
                {"role": "user", "content": self.prompt + str(self.requirements)}
            ],
            n=self.runs,
            temperature=0.7
        )
        self.input_tokens = response.usage.prompt_tokens
        self.output_tokens = response.usage.completion_tokens
        
        for choice in response.choices:
            output_name = self.filename + "_" + str(self.output_counter) + ".txt"
            # Export the response as a .txt file
            path = os.path.join(self.output_fp, output_name)
            with open(path, 'w', encoding='utf8') as text_file:
                print(choice.message.content, file=text_file)
            self.output_counter += 1
            
        # Print a completion messsgae
        print("\n The functions have been generated. \n")
        
        
if __name__ == '__main__':
    function_prompt = "Please generate a set of system functions, in bullet points without numbering, from the following set of equations:"
    llm = PromptChatGPT2Text(prompt=function_prompt, input_fp=r"data\validation_data\PROVE_requirements.txt", output_fp=r"data\PROVE_outputs\PROVE_functions", runs=128)
        
    
