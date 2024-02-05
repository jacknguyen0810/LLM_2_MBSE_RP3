import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util




def main():
    """
    Script for testing the different Sentence-BERT models for complex engineering jargon.

    We want to test the sentiment comparison and speed.

    We will be using a requirement from the James Webb Space Telescope
    (https://spacese.spacegrant.org/uploads/Requirements%20Config/JWST%20Mission%20Requirements%20Document.pdf)
    """

    accuracy = SentenceTransformer("all-mpnet-base-v2")
    speed = SentenceTransformer('all-MiniLM-L6-v2')
    middle_ground = SentenceTransformer('all-miniLm-L12-v2')
    specter = SentenceTransformer('allenai-specter')
    models = {
        'accuracy': accuracy,
        'speed': speed,
        'middle_ground': middle_ground,
        'specter': specter
    }
    
    full_req = "The Observatory shall passively cool the Near-Infrared (NIR) Science Detectors to a temperature of less than or equal to 37K beginning at a time during commissioning that supports NIRCam and NIRSpec commissioning and continuing until the end of the science mission lifetime."
    simple_req = "The Observatory shall passively cool the Near-Infrared Science Detectors to 37K from commission supporting NIRCam and NIRSepc till the end of the mission. "
    basic_req = "The Observatory shall passively cool the detectors throughout the life cycle."
    simplified_dict = {
        'full': full_req,
        'simple': simple_req,
        'basic': basic_req 
    }
    
    results = {
        "Similarity to Complex": [],
        "Similarity to Simple": [],
        "Similarity to Basic": [],
        "Average Similarity": [],
        "Run Time [s]": []
    }
    
    # This is running the analysis
    for name, model in models.items():
        start = time.process_time()
        embeddings_full = model.encode(full_req)
        embeddings_simple = model.encode(list(simplified_dict.values()))
        end = time.process_time()
        run_time = end - start
        cosine_sim = util.cos_sim(embeddings_full, embeddings_simple)
        average = torch.mean(cosine_sim)
        results["Similarity to Complex"].append(cosine_sim.cpu().detach().numpy()[0][0])
        results["Similarity to Simple"].append(cosine_sim.cpu().detach().numpy()[0][1])
        results["Similarity to Basic"].append(cosine_sim.cpu().detach().numpy()[0][2])
        results['Average Similarity'].append(average.cpu().detach().numpy())
        results['Run Time [s]'].append(run_time)
    
    width = 0.15
    model_names = ('all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'all-miniLm-L12-v2', 'allenai-specter')
    x = np.arange(len(model_names))
    multiplier = 0
    
    
    fig, ax = plt.subplots()
    
    for model_type, result in results.items():
        # Round values in results
        rounded = list(np.around(np.array(result), 2))
        offset = width * multiplier
        rects = ax.bar(x + offset, rounded, width, label=model_type)
        ax.bar_label(rects, padding=3, fontsize=7)
        multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Model Comparison Results')
    ax.set_xticks(x + width, model_names)
    ax.legend(loc='upper center', ncols=2)

    plt.show()
    
if __name__ == "__main__":
    main()