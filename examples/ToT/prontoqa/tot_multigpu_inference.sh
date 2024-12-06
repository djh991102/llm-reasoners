base_lm="hf"
hf_path="meta-llama/Meta-Llama-3-8B"
search_algo="beam"

IFS=',' read -ra GPU_ARRAY <<< "0,1,2,3"
NUM_GPUS=${#GPU_ARRAY[@]}

num_sample=4
if [ $num_sample < $NUM_GPUS ]; then
    echo "Number of sample should be larger than number of workers"
    exit 1
fi

# BFS hypterparameter
depth_limit=10
beam_size=10

# Launch process on each GPU in parallel
for i in "${!GPU_ARRAY[@]}"; do
    # Beam Search
    CUDA_VISIBLE_DEVICES=${GPU_ARRAY[$i]} python examples/ToT/prontoqa/tot_inference.py \
    --num_workers ${NUM_GPUS} \
    --worker_idx ${i} \
    --base_lm ${base_lm} \
    --depth_limit ${depth_limit} \
    --hf_path ${hf_path} \
    --temperature 0.8 \
    --search_algo ${search_algo} \
    --beam_size ${beam_size} \
    --batch_size 16 &
done
# Wait for all background processes to complete
wait

# # Combine logs
# cat logs/BeamSearch_*/ > logs/BeamSearch_combined.log