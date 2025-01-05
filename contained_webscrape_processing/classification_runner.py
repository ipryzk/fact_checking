import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class ClassificationChecker:
    def __init__(self, model_name: str, tokenizer_name: str, input_json: str):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.input_json = input_json
        self.corroborate_file = "processing_dir/corroborate.json"
        self.contradict_file = "processing_dir/contradict.json"
        self.pass_count = 0
        self.corroborate_count = 0
        self.contradict_count = 0

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3, ignore_mismatched_sizes=True)

        # Run
        self.load_data()
        self.process_text()
        self.print_counts()
    
    def load_data(self):
        """Load the input JSON file containing the text data."""
        with open(self.input_json, "r") as file:
            self.data = json.load(file)

    def process_text(self):
        """Process each item in the JSON file and classify the text."""
        corroborate_results = []
        contradict_results = []

        for item in self.data:
            text = item.get('text')
            if text:
                result = self.run_model(text)

                # Update counts and add to respective lists
                label = result['label']

                if label == "pass":
                    self.pass_count += 1
                elif label == "corroborate":
                    self.corroborate_count += 1
                    corroborate_results.append(item)
                elif label == "contradict":
                    self.contradict_count += 1
                    contradict_results.append(item)
        
        # Save results to separate JSON files
        self.save_results(self.corroborate_file, corroborate_results)
        self.save_results(self.contradict_file, contradict_results)

    def run_model(self, text: str):
        """Run the fine-tuned model on the input text and return the classification result."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        # Map the class index to label
        label_mapping = {0: "corroborate", 1: "contrast", 2: "pass"}
        predicted_label = label_mapping[predicted_class]

        return {'label': predicted_label}

    def save_results(self, file_name: str, results: list):
        """Save the results (corroborate/contradict) to their respective JSON files."""
        with open(file_name, "w") as file:
            json.dump(results, file, indent=4)

    def print_counts(self):
        """Print the counts of passes, corroborates, and contradicts."""
        print(f"Pass count: {self.pass_count}")
        print(f"Corroborate count: {self.corroborate_count}")
        print(f"Contradict count: {self.contradict_count}")
    
    def return_counts(self):
        """Returns the counts of passes, corroborates, and contradicts."""
        return {"corroborates": self.corroborate_count, "contrasts": self.contradict_count, "passes": self.pass_count}

# Example usage:
# Initialize the fact checker with a model and input JSON file
fact_checker = ClassificationChecker(
    model_name="ipryzk/deberta-large-finetuned", 
    tokenizer_name="microsoft/deberta-large", 
    input_json="processing_dir/retrieved_articles.json"
)

# Load the input data and process the text
fact_checker.load_data()
fact_checker.process_text()

# Print the counts of the labels
fact_checker.print_counts()
