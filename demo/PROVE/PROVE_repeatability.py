from llm_similarity_analysis.repeatability.llm_repeatability_analysis import LLMRepeatabilityAnalysis

def main():
    # Folder containing the inputs 
    data_folder = r"data\PROVE_outputs\PROVE_functions"
    
    # Init the analysis class
    analysis = LLMRepeatabilityAnalysis(data_folder)
    
    # Run analysis
    analysis.run()
    
    # Perform and plot statistical analysis
    analysis.stat_analysis(plot=True)
    
    
    