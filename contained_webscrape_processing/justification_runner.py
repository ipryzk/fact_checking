import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class JustificationChecker:
    def __init__(self, model_name: str, tokenizer_name: str, input_corroborate: str, input_contrast: str):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.input_corroborate = input_corroborate
        self.input_contrast =input_contrast
        self.corroborate_file = "processing_dir/corroborate_justified.json"
        self.contradict_file = "processing_dir/contradict_justified.json"

        # Load the tokenizer and model #ipryzk/model-1b-llama-factcheck-16bit-head
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            self.load_data()
            self.process_text()
        
    def load_data(self):
        """Load the input JSON file containing the text data."""
        with open(self.input_corroborate, "r", encoding="utf-8") as file:
            print("READING JSON!")
            self.corroborate_data = json.load(file)
        with open(self.input_contrast, "r", encoding="utf-8") as file:
            print("READING JSON!")
            self.contrast_data = json.load(file)

    def process_text(self):
        """Process each item in the JSON file and classify the text."""
        corroborate_justifications = []
        contradict_justifications = []
        
        for item in self.corroborate_data:
            text = self.generate_text(item, "corroborate")
            item["justification"] = text
            corroborate_justifications.append(item)
            print("processing")
        
        for item in self.contrast_data:
            text = self.generate_text(item, "contrast")
            item["justification"] = text
            contradict_justifications.append(item)
            print("processing")

        with open(self.corroborate_file, "w") as file:
            json.dump(corroborate_justifications, file, indent=4)
        with open(self.contradict_file, "w") as file:
            json.dump(contradict_justifications, file, indent=4)

        # Save results to separate JSON files
        self.save_results(self.corroborate_file, corroborate_justifications)
        self.save_results(self.contradict_file, contradict_justifications)

    def generate_text(self, input: dict, type: str):
        claim = input['claim']
        text = input['text']

        if type == "corroborate":
            print("starting")
            merged_input = f"<|start_header_id|>user<|end_header_id|>\n\n Task: corroborate the claim using the given text. Claim: {claim} [SEP] Text: {text}. [SEP] Justify:\n\n"
            inputs = self.tokenizer(merged_input, return_tensors="pt").to(self.device)

            outputs = self.model.generate(inputs['input_ids'], max_length=1024, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.92, temperature=0.7)
            "done"
        elif type == "contrast":
            merged_input = f"<|start_header_id|>user<|end_header_id|>\n\n Task: contrast the claim using the given text. Claim: {claim} [SEP] Text: {text}. [SEP] Justify:\n\n"
            inputs = self.tokenizer(merged_input, return_tensors="pt").to(self.device)

            outputs = self.model.generate(inputs['input_ids'], max_length=1024, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.92, temperature=0.7)
            
        # Decode the generated output
        try:
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("[SEP]")[-1].split("Justify:")[-1].strip()
        except:
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
        
    
    def save_results(self, file_name: str, results: list):
        """Save the results (corroborate/contradict) to their respective JSON files."""
        with open(file_name, "w") as file:
            json.dump(results, file, indent=4)
