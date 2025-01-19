import json
from datasets import load_dataset

# Load the GSM8K dataset
dataset = load_dataset("gsm8k", 'main', split="train")

# Initialize an empty dictionary to store the processed data
processed_data = {}

# Process each example in the dataset
for i, example in enumerate(dataset):
    question = example['question']
    cot = example['answer']
    
    # Split the answer into sentences for the chain of thought
    chain_of_thought = cot.split('\n')[:-1]
    answer = cot.split('#### ')[-1]
    # Create the structured example
    structured_example = {
        "test_example": {
            "question": "",
            "query": question,
            "chain_of_thought": chain_of_thought,
            "answer": answer
        }
    }
    
    # Add the structured example to the processed data
    processed_data[f"example_{i+1}"] = structured_example
# Define the output file path
output_file_path = "/home/doyoung/llm-reasoners/examples/CoT/gsm8k/data/ToT_test.json"

# Write the processed data to the JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(processed_data, output_file, indent=4)

print(f"Processed data has been saved to {output_file_path}")