base_lm="vllm"
hf_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
search_algo="beam"

IFS=',' read -ra GPU_ARRAY <<< "0,1,2,3,5,6,7"
NUM_GPUS=${#GPU_ARRAY[@]}
log_dir="/home/jaehyeok/llm-reasoners/logs/gsm8k_cot_ds_test"

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
    --search_algo ${search_algo} \
    --log_dir ${log_dir} \
    --batch_size 5 &
done

# Wait for all background processes to complete
wait