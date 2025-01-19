import json
from datasets import load_dataset
from collections import Counter

# Load the GSM8K dataset
dataset = load_dataset("gsm8k", 'main', split="train")

# Initialize an empty dictionary to store the processed data
processed_data = {}
lengths = []  # List to store lengths of chain_of_thought

# Process each example in the dataset
for i, example in enumerate(dataset):
    question = example['question']
    cot = example['answer']
    
    # Split the answer into sentences for the chain of thought
    chain_of_thought = cot.split('\n')[:-1]
    lengths.append(len(chain_of_thought))  # Store the length
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

# Print statistics
print(f"Average length: {sum(lengths)/len(lengths):.2f}")
print(f"Min length: {min(lengths)}")
print(f"Max length: {max(lengths)}")
print("\nLength distribution:")
counter = Counter(lengths)
for length, count in sorted(counter.items()):
    print(f"Length {length}: {count} examples")

# Rest of your code remains the same...