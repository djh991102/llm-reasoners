from typing import Union, Optional
import warnings
import copy

from transformers import AutoTokenizer
import torch
import numpy as np

from vllm import LLM, SamplingParams

from .. import LanguageModel,GenerateOutput

class VLLMModel(LanguageModel):
    def __init__(self, model_pth, tokenizer_pth, device='cuda:0', max_batch_size=1, max_new_tokens=None, gpu_memory_utilization=0.9, **kwargs):
        super().__init__()
        """
        Initializes a new instance of the `HFModel` class.

        Args:
            model_pth (str): The path to the directory containing the pre-trained model.
            tokenizer_pth (str): The path to the directory containing the pre-trained tokenizer.
            device (str): The device to use for running the model (e.g. "cpu", "cuda").
            max_batch_size (int, optional): The maximum batch size to use for inference. Defaults to 1.
            max_new_tokens (int, optional): The maximum number of new tokens to generate during inference. Defaults to None.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth, lagacy=False, trust_remote_code=True)

        self.model = LLM(
            model=model_pth,
            tokenizer=tokenizer_pth,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size
        self.device = device
    def generate(
            self,
            inputs: list[str],
            max_length: Optional[int] = None,
            max_new_tokens: Optional[int] = None,
            temperature: float = 1.0,
            top_k: int = 10,
            top_p: float = 0.8,
            num_return_sequences: int = 1,
            eos_token_id: Union[None, str, int, list[str, int]] = None,
            output_log_probs: bool = False,
            hide_input=True,
            do_sample=True,
            **kwargs,
        ) -> GenerateOutput:

        # unify eos_token
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        eos_token_id_input = copy.deepcopy(eos_token_id)
        eos_token_id = []

        if temperature == 0.0:
            warnings.warn('temperature=0.0 is equivalent to greedy search, ')
            temperature = 1.0
            top_k = 1 
        if eos_token_id_input is not None:
            if not isinstance(eos_token_id_input, list):
                eos_token_id_input = [eos_token_id_input]
            for token in eos_token_id_input:
                if isinstance(token, str):
                    tokenized = self.tokenizer.encode(token, add_special_tokens=False)
                    if len(tokenized) != 1:
                        warnings.warn(f'the eos_token {repr(token)} is encoded into {tokenized} with length != 1, '
                                    f'using {tokenized[-1]} as the eos_token_id')
                    token = tokenized[-1]
                if isinstance(token, int):
                    eos_token_id.append(token)
                else:
                    warnings.warn(f'the eos_token {repr(token)} is neither str nor int, which is ignored')
        eos_token_id.append(self.tokenizer.eos_token_id)

        generation_config = SamplingParams(
            temperature=temperature,
            stop_token_ids=eos_token_id,
            top_k=top_k,
            top_p=top_p,
            logprobs=5 if output_log_probs else None,
            include_stop_str_in_output=True,
        )
        if max_new_tokens is not None:
            generation_config = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop_token_ids=eos_token_id,
            top_k=top_k,
            top_p=top_p,
            logprobs=5 if output_log_probs else None,
            include_stop_str_in_output=True,
        )
        
        if num_return_sequences > 1:
            assert len(inputs) == 1, 'num_return_sequences > 1 is not supported for multiple inputs'
            inputs = inputs * num_return_sequences
        
        decoded_list = []
        log_prob_list = []
        for start in range(0, len(inputs), self.max_batch_size):
            end = min(start + self.max_batch_size, len(inputs))

            curr_inputs = inputs[start:end]
            with torch.inference_mode():
                generation_output = self.model.generate(
                    prompts=curr_inputs,
                    sampling_params=generation_config,
                    use_tqdm=False,
                )
            log_prob = None
            if output_log_probs:
                log_prob = generation_output.scores
                log_prob_list.extend(log_prob)
            
            if hide_input:
                decoded_list.extend([output.outputs[0].text for output in generation_output])
            else:
                decoded_list.extend([curr_inputs[i] + output.outputs[0].text for i, output in enumerate(generation_output)])
        if not output_log_probs:
            log_prob_list = None

        return GenerateOutput(decoded_list, log_prob_list)

    @torch.no_grad()
    def get_next_token_logits(
        self,
        prompt: Union[str, list[str]],
        candidates: Union[list[str], list[list[str]]]) -> list[np.ndarray]:
        raise Exception("Not Implemented")
    
    @torch.no_grad()
    def get_loglikelihood(self, prefix: str, contents: list[str], **kwargs) -> np.ndarray:
        raise Exception("Not Implemented")