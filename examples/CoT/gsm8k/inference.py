from reasoners.lm import ExLlamaModel
import json
from reasoners.lm.openai_model import OpenAIModel
from reasoners.benchmark import GSM8KEvaluator
from reasoners.lm.hf_model import HFModel
from reasoners.lm.gemini_model import BardCompletionModel
from reasoners.lm.anthropic_model import ClaudeModel
from reasoners.lm import  Llama2Model, Llama3Model
import utils
from typing import Type, Callable, Optional, Literal
import fire
import transformers
import os

class CoTReasoner():
    def __init__(self, base_model, n_sc=1, temperature=0, bs=1):
        assert n_sc == 1 or temperature > 0, \
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs
    def __call__(self, example, prompt=None):
        inputs = prompt["cot"].replace("{QUESTION}", example)
        print(f"example: {example}")
        outputs = []
        do_sample = True
        if self.temperature == 0 and isinstance(self.base_model, HFModel):
            print("Using greedy decoding with HF model. Set do_sample=False")
            self.temperature == 1.0
            do_sample = False
        if isinstance(self.base_model, OpenAIModel) or isinstance(self.base_model, BardCompletionModel) or isinstance(self.base_model, ClaudeModel):
            eos_token_id = []
        elif isinstance(self.base_model.model, transformers.GemmaForCausalLM):
            eos_token_id = [108]
        elif isinstance(self.base_model.model, transformers.MistralForCausalLM) or isinstance(self.base_model.model, transformers.MixtralForCausalLM):
            eos_token_id = [13]
        elif isinstance(self.base_model, Llama2Model):
            eos_token_id = [13]
        elif isinstance(self.base_model, Llama3Model):
            eos_token_id = ["\n\n", ".\n", "\n", ".\n\n"]
        elif self.base_model.model.config.architectures[0] == 'InternLM2ForCausalLM':
            eos_token_id = [364,402,512,756]
        elif self.base_model.model.config.architectures[0] == 'Qwen2ForCausalLM':
            eos_token_id = [198,271,382,624,151645]
        else:
            assert isinstance(self.base_model.model, transformers.LlamaForCausalLM)###need to be modified for other model
            eos_token_id = [13]
        for i in range((self.n_sc - 1) // self.bs + 1):
            local_bs = min(self.bs, self.n_sc - i * self.bs)
            outputs += self.base_model.generate([inputs] * local_bs,
                                            hide_input=True,
                                            do_sample=do_sample,
                                            temperature=self.temperature,
                                            eos_token_id=eos_token_id).text
        outputs= [o.strip() if o.strip().endswith(".") else o.strip() + "." for o in outputs]
        print(f"outputs: {outputs}")
        return outputs
llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
def main(
            model_dir: str = llama_ckpts,
            base_lm: Literal['hf', 'google', 'openai', 'anthropic','exllama',"llama2"] = 'llama2',
            lora_dir=None, mem_map=None, batch_size=1, prompt="examples/CoT/gsm8k/prompts/cot.json",
            hf_path: str = 'meta-llama/Llama-2-13b-hf',
            hf_peft_path: Optional[str] = None,
            hf_quantized: Optional[Literal['awq', 'int8', 'fp4', 'nf4']] = None,
            hf_load_awq_path: Optional[str] = None,
            
            resume=0,
            log_dir: Optional[str] = None,
            temperature=0, n_sc=1, quantized='int8',llama_size=None):

    if base_lm == "openai":
        base_model = OpenAIModel("gpt-4-1106-preview", additional_prompt="ANSWER")
    elif base_lm == "google":
        base_model = BardCompletionModel("gemini-pro", additional_prompt="ANSWER")
    elif base_lm == "anthropic":
        base_model = ClaudeModel("claude-3-opus-20240229", additional_prompt="ANSWER")
    elif base_lm == "hf":
        from reasoners.lm import HFModel
        base_model = HFModel(hf_path, hf_path, max_batch_size=batch_size, max_new_tokens=64,
                                peft_pth=hf_peft_path, quantized=hf_quantized, load_awq_pth=hf_load_awq_path)
    elif base_lm == 'llama2':
        base_model = Llama2Model(model_dir, llama_size, max_batch_size=batch_size)
    elif base_lm == 'llama3':
        base_model = Llama3Model(model_dir, llama_size, max_batch_size=batch_size)
    else:
        raise ValueError(f"Unknown base_lm: {base_lm}")
    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model, temperature=temperature, n_sc=n_sc, bs=batch_size)
    evaluator = GSM8KEvaluator(
                 output_extractor=utils.cot_sc_extractor,
                 answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt, # will update dynamically
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="cot")
    from datetime import datetime
    log_dir =  f'logs/gsm8k_'\
                        f'cot/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)
"""
CUDA_VISIBLE_DEVICES=2 python examples/cot/cot_gsm8k/inference.py \
--model_dir $Gemma_ckpts \ 
"""


