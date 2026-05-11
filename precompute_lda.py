import os
import json
import pickle
import traceback
from utils.text_analyzer import TextAnalyzer
from utils.json_processor import JSONProcessor
from utils.lda import perform_lda_analysis

def get_methods_texts(json_data):
    if isinstance(json_data, list):
        methods_fields = []
        for item in json_data:
            if isinstance(item, dict) and "materials_and_methods" in item:
                methods_fields.append(item["materials_and_methods"])
        return methods_fields
    return []

def main():
    analyzer = TextAnalyzer()
    
    datasets = {
        "PubMed": "attached_assets/methods_pubmed_150.json",
        "PLOS": "attached_assets/methods_plos_1612.json"
    }
    
    out_dir = "attached_assets/precomputed_lda"
    os.makedirs(out_dir, exist_ok=True)
    
    for source, filepath in datasets.items():
        print(f"Loading {source} from {filepath}...")
        try:
            with open(filepath, "r") as f:
                string_data = f.read()
            json_data = JSONProcessor.parse_json(string_data)
            texts = get_methods_texts(json_data)
        except Exception as e:
            print(f"Failed to load {source}: {e}")
            continue
            
        print(f"Found {len(texts)} articles with methods for {source}.")
        if not texts:
            continue
        
        for num_topics in range(3, 16):
            out_file = os.path.join(out_dir, f"{source}_lda_topics_{num_topics}.pkl")
            if os.path.exists(out_file):
                print(f"Skipping {source} with {num_topics} topics (already exists).")
                continue
                
            print(f"Computing LDA for {source} with {num_topics} topics...")
            try:
                html_res, model, dictionary = perform_lda_analysis(analyzer, texts, num_topics=num_topics)
                if html_res and model and dictionary:
                    with open(out_file, "wb") as f:
                        pickle.dump((html_res, model, dictionary), f)
                    print(f"Saved to {out_file}")
                else:
                    print(f"LDA computation returned None for {source} {num_topics} topics.")
            except Exception as e:
                print(f"Error computing LDA for {source} with {num_topics} topics: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    main()
