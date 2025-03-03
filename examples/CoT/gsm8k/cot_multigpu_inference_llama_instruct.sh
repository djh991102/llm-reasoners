base_lm="vllm"
hf_path="meta-llama/Meta-Llama-3-8B-Instruct"

IFS=',' read -ra GPU_ARRAY <<< "4,5,6,7"
NUM_GPUS=${#GPU_ARRAY[@]}
num_shot=4
log_dir="/home/jaehyeok/llm-reasoners/logs/gsm8k_cot_ds_test_llama_instruct_${num_shot}shot"

# BFS hyperparameter

# Launch process on each GPU in parallel
for i in "${!GPU_ARRAY[@]}"; do
    # Beam Search
    CUDA_VISIBLE_DEVICES=${GPU_ARRAY[$i]} python examples/CoT/gsm8k/inference.py \
    --num_workers ${NUM_GPUS} \
    --worker_idx ${i} \
    --base_lm ${base_lm} \
    --hf_path ${hf_path} \
    --temperature 0.0 \
    --gpu_memory_utilization 0.9 \
    --log_dir ${log_dir} \
    --num_shot ${num_shot} \
    --batch_size 5 &
done

# Wait for all background processes to complete
wait