from reasoners.lm import ExLlamaModel
import json
import fire
from typing import Sequence, Any
import json
from tqdm import tqdm
from typing import Type, Callable, Optional, Literal
import os
import examples.CoT.gsm8k.utils as utils
from dataset import GSM8kDataset, GSM8kExample
from reasoners import Reasoner
import torch
import numpy as np
import random
from reasoners import WorldModel, SearchConfig
from reasoners.algorithm import MCTS, BeamSearch, DFS
from reasoners.benchmark import GSM8KEvaluator

GSM8kState = list[str]
GSM8kAction = str

def remove_so_prefix(s):
    if s.startswith('So '):
        return s[3:]
    return s

class GSM8kToTWorldModel(WorldModel[GSM8kState, GSM8kAction, GSM8kExample]):
    def __init__(self) -> None:
        super().__init__()
    
    def init_state(self) -> GSM8kState:
        return []
    
    def step(self, state: GSM8kState, action: GSM8kAction) -> tuple[GSM8kState, dict]:
        return state + [action], {}
    
    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and "The answer is" in state[-1]:
            return True
        return False
    
class GSM8kToTSearchConfig(SearchConfig[GSM8kState, GSM8kAction, GSM8kExample]):
    def __init__(self, base_model, temperature=0.8, add_gold='gold', n_actions=5) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.temperature = temperature
        self.base_model = base_model
        self.add_gold = add_gold
        assert temperature > 0, "Temperature = 0 indicates greedy decoding. There is no point running multiple chains"
    def get_actions(self, state: GSM8kState, test_example_question: str) -> list[GSM8kAction]:
        input_prompt = self.prompt["cot"].replace("{QUESTION}", self.example.test_example.query)
        current_states = "".join([" " + s for s in state])
        input_prompt += current_states
        validate_prompt = '''Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Partial Answer:
Action 1: Today, grove workers will plant total 21 trees total.
Action 2: Today, Grove workers will need to plant 21 trees total.
Do these two actions mean the same thing?
Answer: yes

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Partial Answer: 
Action 1: 3 + 2 gives us 5 cars total.
Action 2: There was initially 3 empty spaces, and 2 cars filled them.
Do these two actions mean the same thing?
Answer: no

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Partial Answer: Altogether they had 74 chocolates initially.
Action 1: 74 minus 35 equals 39 chocolates left.
Action 2: Each lost 17 chocolates, so they ate 34 in total.
Do these two actions mean the same thing?
Answer: no

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Partial Answer: He started with 20 and ended with 12.
Action 1: Jason has given <<20-12=8>>8 lollipops to Denny.
Action 2: Jason gave <<20-12=8>>8 lollipops to Denny.
Do these two actions mean the same thing?
Answer: yes

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Partial Answer: He originally 5 toys. Also he got 2 plus more from mom and dad.
Action 1: He received <<2+2=4>>4 new toys.
Action 2: Shawn received <<2+2=4>>4 more.
Do these two actions mean the same thing?
Answer: yes

Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
Partial Answer: 
Action 1: We started with 9 and installed more over four days.
Action 2: Over those 4 days, we add 5 each day.
Do these two actions mean the same thing?
Answer: no

Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
Partial Answer: He started with 58 balls. He lost 23 on Tuesday which makes 58 - 23 = 35.
Action 1: On Wednesday, he lost 2 more, so he has <<35-2=33>>33.
Action 2: The amount of balls he lost on Wednesday is 23 + 2 = 25.
Do these two actions mean the same thing?
Answer: no

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Partial Answer: She began with 23 dollars.
Action 1: She spent 5 times 3 dollars, which is 15 dollars.
Action 2: She bought 5 bagels, each costing 3 dollars, so she spent 5 x 3 = 15 dollars.
Do these two actions mean the same thing?
Answer: yes'''
        #eos_token_id=29889
        eos_token_id=[".\n"]
        ret = []
        if self.add_gold == "gold":
            curr_len = len(state)
            gold_trajectory = self.example.test_example.chain_of_thought
            if curr_len <= len(gold_trajectory):
                if curr_len == len(gold_trajectory) and gold_trajectory == state:
                    gold_action = f"The answer is {self.example.test_example.answer.lower()}."
                    if gold_action not in ret:
                        print(f"ADDED {gold_action}")
                        ret.append(gold_action)
                elif gold_trajectory[:curr_len] == state:
                    gold_action = gold_trajectory[curr_len]
                    if gold_action not in ret:
                        print(f"ADDED {gold_action}")
                        ret.append(gold_action)

        output = self.base_model.generate([input_prompt] * (self.n_actions-len(ret)), eos_token_id=eos_token_id, hide_input=True, temperature=self.temperature, do_sample=True).text
        
        for o in output:
            if "." in o:
                ret.append(o.strip()[:o.strip().index(".")+1])
        ret = list(dict.fromkeys(ret).keys())
                        

        # deduplicate only if base model predicts they are the same
        filtered = []
        n = len(ret)
        # print(f"actions before filtering: {ret}: {n} actions")

        # Fill the equivalence matrix more efficiently
        for i in range(n):
            if i not in filtered:  # Skip if this index has been marked as equivalent to a previous one
                for j in range(i+1, n):  # Only check items after i
                    if j not in filtered:  # Skip if this index has been marked as equivalent to a previous one
                        comparison_prompt = f"{validate_prompt}\n\nQuestion: {self.example.test_example.query}\nPartial Answer: {current_states}\nAction 1: {ret[i]}\nAnswer 2: {ret[j]}\nDo these two actions mean the same thing?\nAnswer: "
                        response = self.base_model.generate([comparison_prompt], temperature=0, do_sample=False).text[0].lower()
                        if "yes" in response:
                            filtered.append(j)  # Mark j as equivalent to i
                
            # if i not in filtered:  # If i wasn't marked as equivalent to any previous item
            #     filtered.append(i)
        # Convert indices back to actual statements, taking only unique items
        dedup_filtered = [ret[i] for i in range(n) if i not in filtered]
        ret = dedup_filtered

        # EDITED: REMOVE MODEL HALLUCINATION ===
        filtered_ret = []
        for item in ret:
            ### HEURISTICS ###
            if item.startswith('Q.') or item.startswith('A.') or not item.endswith('.') or item.replace('.','').replace(',','').replace('%','').replace(' ','').isnumeric():
                continue
            if item.replace('.','').replace(',','').replace('%','').replace(' ','')=='':
                continue
            if "Q:" in item or item.strip() == "<<" or item.strip() == ">>":
                continue
            # if len(item) < 5:
            #     continue
            filtered_ret.append(item)
        print(f"actions after filtering: {filtered_ret} ({len(filtered_ret)} actions)")
        # EOL ===

        return filtered_ret

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        return 0, {}
    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        return 0, {}
llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
def main(
           model_dir: str = llama_ckpts,
           base_lm: Literal[ 'llama2',' exllama', 'llama3']  = 'llama2',
           llama_size = "7B",
           batch_size = 4,
           prompt="examples/CoT/gsm8k/prompts/cot.json", 
           hf_path: str = 'meta-llama/Llama-2-13b-hf',
           hf_peft_path: Optional[str] = None,
           hf_quantized: Optional[Literal['awq', 'int8', 'fp4', 'nf4']] = None,
           hf_load_awq_path: Optional[str] = None,
           search_algo: str = "beam",
           num_workers:int = 1,
           worker_idx:int = 0,
           num_sample_per_worker:int = -1, # Debugging purpose
           depth_limit: int = 6,
           log_dir: Optional[str] = None,
           add_gold: str = "gold",
           temperature: float = 0.8,
           mem_map: str = [16, 22],
           gpu_memory_utilization: float=0.9,
           **search_algo_params):

    if search_algo == "beam":
        search_algo_params |= {"max_depth": depth_limit}
    elif search_algo == "dfs":
        search_algo_params |= {"depth": depth_limit}
    else:
        print("Unknown search algorithm", search_algo)
        raise NotImplementedError
    

    def bfs_pronto_extractor(algo_output):
        # if torch.distributed.is_initialized():
        #     torch.distributed.barrier()
        # to make sure the plan is saved before evaluation in multi-process setting
        try:
            answer =algo_output.terminal_node.state[-1].replace("The answer is ", "").replace(".", "").replace(' ', '')
            answer = answer.replace("So ", "")
            return answer

        except Exception as e:
            print("Error in output extraction,", e)
            return ""
    
    def dfs_bw_extractor(algo_output):
        # if torch.distributed.is_initialized():
        #     torch.distributed.barrier()
        # to make sure the plan is saved before evaluation in multi-process setting
        try:
            answer = algo_output.terminal_state[-1].replace("The answer is ", "").replace(".", "").replace(' ', '')
            answer = answer.replace("So ", "")
            return answer

        except Exception as e:
            print("Error in output extraction,", e)
            return ""

    if base_lm in ['llama2', 'llama3']:    
        import torch
        import torch.backends.cudnn
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

    if base_lm == 'llama2':
        from reasoners.lm import Llama2Model
        base_model = Llama2Model(model_dir, llama_size, max_batch_size=batch_size)
    elif base_lm == 'llama3':
        from reasoners.lm import Llama3Model
        base_model = Llama3Model(model_dir, llama_size, max_batch_size=batch_size)
    elif base_lm == 'hf':
        from reasoners.lm import HFModel
        base_model = HFModel(hf_path, hf_path, max_batch_size=batch_size, max_new_tokens=64,
                                peft_pth=hf_peft_path, quantized=hf_quantized, load_awq_pth=hf_load_awq_path)
    elif base_lm == 'vllm':
        from reasoners.lm import VLLMModel
        base_model = VLLMModel(hf_path, hf_path, max_batch_size=batch_size, max_new_tokens=64,
                                gpu_memory_utilization=gpu_memory_utilization)
    else:
        from reasoners.lm import ExLlamaModel  # Maybe other transformer models also support
        base_model = ExLlamaModel(model_dir, 
                                lora_dir=None, 
                                device=torch.device("cuda:0"), 
                                max_batch_size=1, 
                                max_new_tokens=200, 
                                max_seq_length=2048,
                                mem_map=mem_map)

    world_model = GSM8kToTWorldModel()
    search_config = GSM8kToTSearchConfig(base_model=base_model, temperature=temperature, add_gold=add_gold, n_actions=search_algo_params["beam_size"])
    
    output_extractor = dfs_bw_extractor if search_algo == "dfs" else bfs_pronto_extractor
    if search_algo == "dfs":
        search_algo = DFS(**search_algo_params)
    elif search_algo == "beam":
        search_algo = BeamSearch(**search_algo_params)
    else:
        raise NotImplementedError
   
    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = Reasoner(world_model=world_model, search_config=search_config, search_algo=search_algo)


    evaluator = GSM8KEvaluator(
        init_prompt=prompt,
        sample_prompt_type="cot",
        disable_log=False,
        disable_tqdm=False,
        dataset = GSM8kDataset.from_file(
            'examples/CoT/gsm8k/data/ToT_train.json'
        ),
        output_extractor=output_extractor,
        answer_extractor=lambda x: x.test_example.answer
    )

    full_log_path = os.path.join(log_dir, "algo_output")
    if not os.path.exists(full_log_path):
        os.makedirs(full_log_path)

    files = [f for f in os.listdir(full_log_path) if os.path.isfile(os.path.join(full_log_path, f))]
    pass_idx = []
    for file in files:
        log_num = int(file.split(".pkl")[0].strip())
        pass_idx.append(log_num-1)

    num_examples = len(evaluator.full_dataset)-1
    num = int(num_examples / num_workers)
    rmd = num_examples % num_workers

    if rmd != 0 and rmd > worker_idx:
        num_sample = num + 1
    elif rmd != 0:
        num_sample = num
    else:
        num_sample = num

    if rmd == 0:
        resume = num*worker_idx
    else:
        resume = num*worker_idx+(rmd if rmd <= worker_idx else worker_idx)

    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir, num_sample=num_sample_per_worker if num_sample_per_worker != -1 else num_sample, pass_idx = pass_idx)
    print(accuracy)

if __name__ == '__main__':
    import os
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    fire.Fire(main)

