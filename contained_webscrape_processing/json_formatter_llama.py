import json

# Function to add llama_input field
def add_llama_input(data):
    for category in ['corroborate', 'contrast']:
        for item in data.get(category, []):
            classification = "corroborate" if category == 'corroborate' else "contrast"
            item['llama_input'] = f"Task: {classification} the claim using the given text. Claim: {item['claim']}. [sep] Text: {item['text']}. [sep] Justify:"

# Function to read and write the JSON data
def process_json(input_file, output_file):
    # Load the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Add llama_input to the original data
    add_llama_input(data)
    
    # Save the updated data to the output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"JSON file with llama_input added has been saved to {output_file}.")

# Example usage
input_file = 'processing_dir/file_receive.json'  # Input JSON file
output_file = 'processing_dir/formatted_claims.json'  # Output JSON file

# Process the input file and save the result
process_json(input_file, output_file)
