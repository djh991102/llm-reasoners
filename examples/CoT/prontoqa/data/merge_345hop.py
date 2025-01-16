import json

IC_FIXED = [
{
   "question": "Each lepidopteran is an insect. Each arthropod is a protostome. Every animal is multicellular. Protostomes are invertebrates. Each whale is bony. Each painted lady is a butterfly. Invertebrates are animals. Butterflies are lepidopterans. Every insect is six-legged. Every insect is an arthropod. Arthropods are not bony. Sally is a painted lady.",
   "query": "True or false: Sally is not bony.",
   "chain_of_thought": [
    "Sally is a painted lady.",
    "Each painted lady is a butterfly.",
    "Sally is a butterfly.",
    "Butterflies are lepidopterans.",
    "Sally is a lepidopteran.",
    "Each lepidopteran is an insect.",
    "Sally is an insect.",
    "Every insect is an arthropod.",
    "Sally is an arthropod.",
    "Arthropods are not bony.",
    "Sally is not bony."
   ],
   "answer": "True"
},
{
   "question": "Prime numbers are natural numbers. Every Mersenne prime is not composite. Imaginary numbers are not real. Every real number is a number. Natural numbers are integers. Every real number is real. Every Mersenne prime is a prime number. Natural numbers are positive. Prime numbers are not composite. Integers are real numbers. 127 is a Mersenne prime.",
   "query": "True or false: 127 is not real.",
   "chain_of_thought": [
    "127 is a Mersenne prime.",
    "Every Mersenne prime is a prime number.",
    "127 is a prime number.",
    "Prime numbers are natural numbers.",
    "127 is a natural number.",
    "Natural numbers are integers.",
    "127 is an integer.",
    "Integers are real numbers.",
    "127 is a real number.",
    "Every real number is real.",
    "127 is real."
   ],
   "answer": "False"
},
{
   "question": "Lepidopterans are insects. Every animal is multicellular. Each insect is an arthropod. Each invertebrate is an animal. Insects are six-legged. Arthropods are small. Arthropods are invertebrates. Each butterfly is a lepidopteran. Whales are not small. Polly is a lepidopteran.",
   "query": "True or false: Polly is not small.",
   "chain_of_thought": [
    "Polly is a lepidopteran.",
    "Lepidopterans are insects.",
    "Polly is an insect.",
    "Each insect is an arthropod.",
    "Polly is an arthropod.",
    "Arthropods are small.",
    "Polly is small."
   ],
   "answer": "False"
},
{
   "question": "Every cat is a feline. Mammals are vertebrates. Bilaterians are animals. Vertebrates are chordates. Carnivores are mammals. Mammals are not cold-blooded. Each chordate is a bilaterian. Every feline is a carnivore. Snakes are cold-blooded. Animals are not unicellular. Every carnivore is not herbivorous. Fae is a cat.",
   "query": "True or false: Fae is not cold-blooded.",
   "chain_of_thought": [
    "Fae is a cat.",
    "Every cat is a feline.",
    "Fae is a feline.",
    "Every feline is a carnivore.",
    "Fae is a carnivore.",
    "Carnivores are mammals.",
    "Fae is a mammal.",
    "Mammals are not cold-blooded.",
    "Fae is not cold-blooded."
   ],
   "answer": "True"
},
{
   "question": "Prime numbers are prime. Real numbers are numbers. Every integer is a real number. Real numbers are not imaginary. Mersenne primes are prime numbers. Complex numbers are imaginary. Each prime number is a natural number. Natural numbers are positive. Each Mersenne prime is prime. Each natural number is an integer. 7 is a prime number.",
   "query": "True or false: 7 is imaginary.",
   "chain_of_thought": [
    "7 is a prime number.",
    "Each prime number is a natural number.",
    "7 is a natural number.",
    "Each natural number is an integer.",
    "7 is an integer.",
    "Every integer is a real number.",
    "7 is a real number.",
    "Real numbers are not imaginary.",
    "7 is not imaginary."
   ],
   "answer": "False"
},
{
   "question": "Spiders are not six-legged. Insects are six-legged. Insects are arthropods. Every animal is not unicellular. Invertebrates are animals. Lepidopterans are insects. Every arthropod is segmented. Arthropods are invertebrates. Every butterfly is a lepidopteran. Stella is a butterfly.",
   "query": "True or false: Stella is six-legged.",
   "chain_of_thought": [
    "Stella is a butterfly.",
    "Every butterfly is a lepidopteran.",
    "Stella is a lepidopteran.",
    "Lepidopterans are insects.",
    "Stella is an insect.",
    "Insects are six-legged.",
    "Stella is six-legged."
   ],
   "answer": "True"
},
{
   "question": "Each natural number is not negative. Prime numbers are not composite. Mersenne primes are not composite. Real numbers are real. Real numbers are numbers. Mersenne primes are prime numbers. Integers are real numbers. Each imaginary number is not real. Every natural number is an integer. Each prime number is a natural number. 31 is a Mersenne prime.",
   "query": "True or false: 31 is real.",
   "chain_of_thought": [
    "31 is a Mersenne prime.",
    "Mersenne primes are prime numbers.",
    "31 is a prime number.",
    "Each prime number is a natural number.",
    "31 is a natural number.",
    "Every natural number is an integer.",
    "31 is an integer.",
    "Integers are real numbers.",
    "31 is a real number.",
    "Real numbers are real.",
    "31 is real."
   ],
   "answer": "True"
},
{
   "question": "Mammals are vertebrates. Carnivores are mammals. Bilaterians are animals. Vertebrates are chordates. Carnivores are not herbivorous. Tabbies are cats. Every feline is a carnivore. Chordates are bilaterians. Animals are multicellular. Mammals are warm-blooded. Snakes are not warm-blooded. Cats are felines. Sam is a tabby.",
   "query": "True or false: Sam is warm-blooded.",
   "chain_of_thought": [
    "Sam is a tabby.",
    "Tabbies are cats.",
    "Sam is a cat.",
    "Cats are felines.",
    "Sam is a feline.",
    "Every feline is a carnivore.",
    "Sam is a carnivore.",
    "Carnivores are mammals.",
    "Sam is a mammal.",
    "Mammals are warm-blooded.",
    "Sam is warm-blooded."
   ],
   "answer": "True"
},
]

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    f.close()
    return data

def merge_data(data1, data2):
    dedup = []

    for k, v in data1.items():
        dedup.append(v["test_example"])
    
    for k, v in data2.items():
        if v["test_example"] in dedup:
            continue
        dedup.append(v["test_example"])

    ret = {}
    for i, item in enumerate(dedup):
        ret[f"example{i+1}"] = {}
        for ic_ex_idx, ic_ex in enumerate(IC_FIXED):
            ret[f"example{i+1}"][f"in_context_example{ic_ex_idx}"] = ic_ex
        ret[f"example{i+1}"]["test_example"] = item
    return ret

if __name__ == "__main__":
    data1 = load_data("examples/CoT/prontoqa/data/345hop_random_true.json")
    data2 = load_data("examples/CoT/prontoqa/data/generated_ood_train.json")
    ret = merge_data(data1, data2)
    for k,v in ret.items():
        if "Stella is segmented." in v["test_example"]["chain_of_thought"]:
            print(v["test_example"])
            input()

    out_file_path = "examples/CoT/prontoqa/data/merged_345hop_random_true.json"
    with open(out_file_path, 'w') as f:
        json.dump(ret, f)
    f.close()