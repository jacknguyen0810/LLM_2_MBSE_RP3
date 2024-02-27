import os
from llm_similarity_analysis.text_gen.prompt_ChatGPT_api import PromptChatGPT2Text

def main():
    # Get the requirements text file
    PROVE_req = r"data\validation_data\PROVE_requirements.txt"
    
    # Prompt the GPT API to generate a list of system functions if it does not exist already
    PROVE_func = r"data\PROVE_outputs\PROVE_functions\llm_output.txt_1.txt"
    
    
    
    
    