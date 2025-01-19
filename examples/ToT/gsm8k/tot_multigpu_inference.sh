base_lm="hf"
hf_path="meta-llama/Meta-Llama-3-8B"
search_algo="beam"

IFS=',' read -ra GPU_ARRAY <<< "0,1,2,3,4,5,6,7"
NUM_GPUS=${#GPU_ARRAY[@]}

log_dir="logs/gsm8k_generated_OOD_BeamSearch"

# BFS hyperparameter
depth_limit=13
beam_size=5

# Launch process on each GPU in parallel
for i in "${!GPU_ARRAY[@]}"; do
    # Beam Search
    CUDA_VISIBLE_DEVICES=${GPU_ARRAY[$i]} python examples/ToT/gsm8k/tot_inference.py \
    --num_workers ${NUM_GPUS} \
    --worker_idx ${i} \
    --base_lm ${base_lm} \
    --depth_limit ${depth_limit} \
    --hf_path ${hf_path} \
    --temperature 0.8 \
    --search_algo ${search_algo} \
    --beam_size ${beam_size} \
    --log_dir ${log_dir} \
    --add_gold gold \
    --batch_size 32 &
done

# Wait for all background processes to complete
wait