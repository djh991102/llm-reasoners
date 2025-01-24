import os
import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    f.close()
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    f.close()

def get_valid_example_pool(example_pool_path="/home/jaehyeok/llm-reasoners/examples/CoT/prontoqa/data/example_pool.json"):
    example_pool = load_json(example_pool_path)['example_pool']

    valid_pool = []
    for item in example_pool:
        if not is_ood_data(item):
            valid_pool.append(item)
    return valid_pool


def is_ood_data(item):
    if " and " in item['question'] or " or " in item['question']:
        return True
    
    for step in item['chain_of_thought']:
        if " and " in step and " or " in step:
            return True
    return  False

def count_num_hops(data):
    num_hops_dict = {}
    for item in data:
        num_hops = len(item['chain_of_thought'])
        num_hops_dict[num_hops] = num_hops_dict.get(num_hops, 0) + 1
    return num_hops_dict

def get_few_shot_examples(few_shots_path="/home/jaehyeok/llm-reasoners/examples/CoT/prontoqa/prompts/cot.json"):
    few_shot_examples = load_json(few_shots_path)['cot_pool']
    ret = []
    for item in few_shot_examples:
        question, query, chain_of_thought, answer = parse_example(item)
        ret.append({
            'question': question,
            'query': query,
            'chain_of_thought': chain_of_thought,
            'answer': answer,
        })
    return ret

def parse_example(example):
    splitted = example.split("A:")
    question = splitted[0].split("Q:")[-1].strip()
    query_idx = question.index("True or false:")
    query = question[query_idx:].strip()
    question = question[:query_idx].strip()

    answer = splitted[-1].strip()
    chain_of_thought = [x.strip()+"." for x in answer.split('.') if x.strip() != '']
    answer = "True" if "true" in chain_of_thought[-1] else "False"
    return question, query, chain_of_thought[:-1], answer

def deduplication(data, dedup_targets):
    remove_index = []
    for i, item in enumerate(data):
        for dedup_target in dedup_targets:
            if item['question'] == dedup_target['question'] and \
                item['query'] == dedup_target['query'] and \
                 item['chain_of_thought'] == dedup_target['chain_of_thought'] and \
                    item['answer'] == dedup_target['answer']:
                remove_index.append(i)
                break
    for idx in remove_index:
        data.pop(idx)
    return data

def get_1hops_examples(file_path="/home/jaehyeok/llm-reasoners/examples/CoT/prontoqa/data/longface_prontoqa_train.json"):
    data = load_json(file_path)
    ret = []
    for k, v in data.items():
        ret.append(v['test_example'])
    return ret

if __name__ == '__main__':
    valid_pool = get_valid_example_pool()
    num_hops_dict = count_num_hops(valid_pool)

    # Few shot examples dedup
    few_shot_examples = get_few_shot_examples()
    final_pool = deduplication(data=valid_pool, dedup_targets=few_shot_examples)

    num_hops_examples_dict = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }

    for item in final_pool:
        num_hops = len(item['chain_of_thought'])//2
        if num_hops == 3:
            num_hops_examples_dict[3].append(item)
        elif num_hops == 4:
            num_hops_examples_dict[4].append(item)
        elif num_hops == 5:
            num_hops_examples_dict[5].append(item)
        else:
            raise Exception("Invalid number of hops within the example pools")
    
    # Generate 2 hops examples

    import numpy as np
    import random
    random.seed(42)

    random.shuffle(num_hops_examples_dict[3])
    num_hops_examples_dict[2] = num_hops_examples_dict[3][:len(num_hops_examples_dict[3])//2]
    num_hops_examples_dict[3] = num_hops_examples_dict[3][len(num_hops_examples_dict[3])//2:]
    
    for item in num_hops_examples_dict[2]:
        initial_query = item['chain_of_thought'][0]
        target_query = item['chain_of_thought'][2]

        assert initial_query in item['question']
        item['question'] = item['question'].replace(initial_query, target_query)
        item['chain_of_thought'] = item['chain_of_thought'][2:]

        assert initial_query not in item['question'] and target_query in item['question']
    
    # Get 1 hops examples
    num_hops_examples_dict[1] = get_1hops_examples()
    
    num_samples_per_hops = 900
    ret = []
    for k, v in num_hops_examples_dict.items():
        num_hops_examples_dict[k] = random.sample(v, k=num_samples_per_hops)
        ret += num_hops_examples_dict[k]
    
    random.shuffle(ret)
    test_ratio = 0.1
    split_idx = int(len(ret)*test_ratio)
    test_set = ret[:split_idx]
    train_set = ret[split_idx:]

    print(f"Train set distribution: {count_num_hops(train_set)}, Length: {len(train_set)}")
    print(f"Train set distribution: {count_num_hops(test_set)}, Length: {len(test_set)}")

    # Convert format into llm-reasoners compatible
    train_set_final = {}
    test_set_final = {}
    for i, item in enumerate(train_set):
        item_key = f"example{i+1}"
        train_set_final[item_key] = {
            "test_example": item
        }
    for i, item in enumerate(test_set):
        item_key = f"example{i+1}"
        test_set_final[item_key] = {
            "test_example": item
        }

    train_set_out_file_name="/home/jaehyeok/llm-reasoners/examples/CoT/prontoqa/data/1-5hops_train_set.json"
    test_set_out_file_name="/home/jaehyeok/llm-reasoners/examples/CoT/prontoqa/data/1-5hops_test_set.json"

    save_json(train_set_final, train_set_out_file_name)
    save_json(test_set_final, test_set_out_file_name)