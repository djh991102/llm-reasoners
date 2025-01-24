from reasoners.lm import ExLlamaModel
import json
import fire
from typing import Sequence, Any
import json
from tqdm import tqdm
from typing import Type, Callable, Optional, Literal
import os

from dataset import ProntoQADataset, ProntoQAExample
from reasoners import Reasoner
import torch
import numpy as np
import random
from reasoners import WorldModel, SearchConfig
from reasoners.algorithm import BeamSearch, DFS
from reasoners.benchmark import ProntoQAEvaluatorFinal

ProntoQAState = list[str]
ProntoQAAction = str

def remove_so_prefix(s):
    if s.startswith('So '):
        return s[3:]
    return s

class ProntoQAToTWorldModel(WorldModel[ProntoQAState, ProntoQAAction, ProntoQAExample]):
    def __init__(self) -> None:
        super().__init__()
    
    def init_state(self) -> ProntoQAState:
        return []
    
    def step(self, state: ProntoQAState, action: ProntoQAAction) -> tuple[ProntoQAState, dict]:
        return state + [action], {}
    
    def is_terminal(self, state: ProntoQAState) -> bool:
        if len(state) > 0 and "The answer is" in state[-1]:
            return True
        return False
    
class ProntoQAToTSearchConfig(SearchConfig[ProntoQAState, ProntoQAAction, ProntoQAExample]):
    def __init__(self, base_model, n_actions=5, add_gold='gold',temperature=0.8) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.temperature = temperature
        self.base_model = base_model
        self.add_gold = add_gold
        assert temperature > 0, "Temperature = 0 indicates greedy decoding. There is no point running multiple chains"
    def get_actions(self, state: ProntoQAState, test_example_question: str) -> list[ProntoQAAction]:
        input_prompt = self.prompt['cot'].replace("{QUESTION}", self.example.test_example.question).replace("{QUERY}", self.example.test_example.query)
        input_prompt += "".join([" " + s for s in state])
        
        eos_token_id=["."]
        ret = []
        
        output = self.base_model.generate([input_prompt] * self.n_actions, eos_token_id=eos_token_id, hide_input=True, temperature=self.temperature, do_sample=True).text

        for o in output:
            if "." in o:
                ret.append(o.strip()[:o.strip().index(".")+1])
        
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
        
        # deduplicate
        ret = list(dict.fromkeys(ret).keys())

        # EDITED: REMOVE MODEL HALLUCINATION ===
        filtered_ret = []
        for item in ret:
            if len(state) % 2 == 1:
                if "The answer is " not in item:
                    if item in test_example_question:
                        filtered_ret.append(item)
                else:
                    filtered_ret.append(item)
            else:
                ### HEURISTICS ###
                if "1." in item or "2." in item or "3." in item or "4." in item or "5." in item:
                    continue
                filtered_ret.append(item)
        # EOL ===

        return filtered_ret

    def fast_reward(self, state: ProntoQAState, action: ProntoQAAction) -> tuple[float, dict]:
        return 0, {}
    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        return 0, {}
llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
def main(
           model_dir: str = llama_ckpts,
           base_lm: Literal[ 'llama2',' exllama', 'llama3']  = 'llama2',
           llama_size = "7B",
           batch_size = 4,
           prompt="examples/CoT/prontoqa/prompts/cot.json",
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
    if log_dir is not None and not os.path.exists(log_dir):
        os.makedirs(log_dir)
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
            answer = "\n".join(algo_output.terminal_node.state[2::2])
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
            answer = "\n".join(algo_output.terminal_state[2::2])
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

    world_model = ProntoQAToTWorldModel()
    search_config = ProntoQAToTSearchConfig(base_model=base_model, temperature=temperature, add_gold=add_gold, n_actions=search_algo_params["beam_size"])
    
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
    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=prompt,
        sample_prompt_type="cot",
        disable_log=False,
        disable_tqdm=False, 
        dataset = ProntoQADataset.from_file(
            'examples/CoT/prontoqa/data/1-5hops_train_set.json'
        ),
        output_extractor=output_extractor,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2])
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

# CUDA_VISIBLE_DEVICES=0 python examples/tot/prontoqa/inference_tot.py --depth_limit 10 --model_dir $LLAMA2_CKPTS --beam_size 10 --temperature 0.8 --reward_aggregator mean --search_algo beam > debug_bfs.log

# python examples/ToT/prontoqa/tot_inference.py --base_lm hf --depth_limit 13 --hf_path meta-llama/Meta-Llama-3-8B --temperature 0.8 --search_algo beam --beam_size=3 --batch_size=16 --log_dir="logs/prontoqa_Beamsearch"

# python examples/tot/prontoqa/tot_inference.py --depth_limit 10 --model_dir /data/yi/Llama-2-70B-GPTQ/ --total_states 10 --temperature 0.8 --search_algo dfs --max_per_state 3 > debug_dfs.log
