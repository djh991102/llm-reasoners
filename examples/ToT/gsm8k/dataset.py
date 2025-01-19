from dataclasses import dataclass, field
from typing import List, Dict, Iterator, Mapping

import json
from dataclasses import dataclass, field
from typing import List, Dict, Iterator, Union


@dataclass
class GSM8kProblem:
    question: str
    query: str
    chain_of_thought: List[str]
    answer: str


@dataclass
class GSM8kExample:
    in_context_examples: Mapping[str, GSM8kProblem]
    test_example: GSM8kProblem = field(default=None)
    """
    GSM8kProblem(question='Every cat is a feline. Mammals are vertebrates. Bilaterians are animals. Vertebrates are chordates. Carnivores are mammals. Mammals are not cold-blooded. Each chordate is a bilaterian. Every feline is a carnivore. Snakes are cold-blooded. Animals are not unicellular. Every carnivore is not herbivorous. Fae is a cat.', query='True or false: Fae is not cold-blooded.', chain_of_thought=['Fae is a cat.', 'Every cat is a feline.', 'Fae is a feline.', 'Every feline is a carnivore.', 'Fae is a carnivore.', 'Carnivores are mammals.', 'Fae is a mammal.', 'Mammals are not cold-blooded.', 'Fae is not cold-blooded.'], answer='True')"""



@dataclass
class GSM8kDataset:
    examples: Dict[str, GSM8kExample] = field(default_factory=dict)
    # "example1": ..., "example2": ..., ...

    @classmethod
    def from_file(cls, file_path: str) -> 'GSM8kDataset':
        instance = cls()
        with open(file_path, 'r') as f:
            raw_data = json.load(f)

        for example_key, example_value in raw_data.items():

            all_examples = {
                k: GSM8kProblem(
                    question=e["question"],
                    query=e["query"],
                    chain_of_thought=e["chain_of_thought"],
                    answer=e["answer"]
                ) for k, e in example_value.items()
            }

            test_example = all_examples.pop('test_example', None)
            in_context_examples = []

            instance.examples[example_key] = GSM8kExample(in_context_examples, test_example)

        return instance

    def __iter__(self) -> Iterator[Union[GSM8kProblem, GSM8kExample]]:
        return iter(self.examples.values())


# Sample usage
if __name__ == "__main__":
    pronto_qa_dataset = GSM8kDataset.from_file('/home/doyoung/llm-reasoners/examples/CoT/gsm8k/data/ToT_train.json')

    for i, example in enumerate(pronto_qa_dataset):
        print(example)