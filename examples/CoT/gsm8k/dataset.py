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
    GSM8kProblem(question='Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', query='', chain_of_thought=['Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May.', 'Altogether, she sold 48 + 24 = 72 clips.'], answer='72')"""

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
    pronto_qa_dataset = GSM8kDataset.from_file('/Users/xiyan/Downloads/345hop_random_true.json')

    for example in pronto_qa_dataset:
        print(example)