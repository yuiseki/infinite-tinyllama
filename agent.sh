# Autonomous Agent that train LLMs written by shell script
#
# Description
# This script is the main script for the autonomous agent. It is responsible for the followings
# 1. Monitoring the state of the computational resources
# 2. Making decisions based on the state of the computational resources
#   2.1. Keep waiting - Continue to wait until computing resources are available
#   2.2. Search task - Search task on the available computing resources
#   2.3. Generate recipes - Generate recipes for the new LLM based on the state of the computational resources and the found task

# Get GPU stat by gpustat command
GPU_STAT=$(gpustat -cp --no-color | tail -n +2)
# Example of GPU_STAT
# [0] NVIDIA RTX A5000 | 48'C,  81 % |  3279 / 24564 MB | ollama/8919(526M) python3/486829(2434M)
# [1] NVIDIA RTX A5000 | 47'C,   0 % | 11389 / 24564 MB | ollama/8919(526M) python3/486829(10544M)
# [2] NVIDIA RTX A5000 | 48'C,   0 % | 11569 / 24564 MB | ollama/8919(526M) python3/486829(10724M)
# [3] NVIDIA RTX A5000 | 56'C,  19 % | 16113 / 24564 MB | ollama/8919(526M) python3/486829(15268M)
# [4] NVIDIA RTX A5000 | 32'C,   0 % |   845 / 24564 MB | ollama/8919(526M)
# [5] NVIDIA RTX A5000 | 30'C,   0 % |   845 / 24564 MB | ollama/8919(526M)
# [6] NVIDIA RTX A5000 | 31'C,   0 % |   845 / 24564 MB | ollama/8919(526M)
# [7] NVIDIA RTX A5000 | 31'C,   0 % |   845 / 24564 MB | ollama/8919(526M)
# End of example of GPU_STAT

# GPU_NUM is rows of GPU_STAT
GPU_NUM=$(echo "$GPU_STAT" | wc -l)
echo "GPU_NUM: $GPU_NUM"

# GPU_MEM_USAGE is the memory usage of each GPU
# GPU_MEM_USAGE is an array of GPU_NUM elements

GPU_MEM_USAGE=()
for i in $(seq $GPU_NUM); do
  # For example, GPU_MEM_USAGE[0] is the memory usage of the first GPU
  # extract string as 3279 / 24564 MB from [0] NVIDIA RTX A5000 | 48'C,  81 % |  3279 / 24564 MB | ollama/8919(526M) python3/486829(2434M)
  TMP_MEM_USAGE=$(echo "$GPU_STAT" | sed -n "${i}p" | awk '{print $10}')
  TMP_MEM_TOTAL=$(echo "$GPU_STAT" | sed -n "${i}p" | awk '{print $12}')
  # MEM_USAGE is 3279 / 24564 MB, calculate the TMP_MEM_USAGE / TMP_MEM_TOTAL * 100
  MEM_USAGE=$(echo "scale=2; $TMP_MEM_USAGE / $TMP_MEM_TOTAL * 100" | bc)
  echo "MEM_USAGE: $MEM_USAGE"
  GPU_MEM_USAGE+=($MEM_USAGE)
done
# echo "GPU_MEM_USAGE: ${GPU_MEM_USAGE[@]}"

# If the memory usage of the GPU is less than 5%, the agent will search for new task
# How many GPUs are available? and what is index of the available GPU?
AVAILABLE_GPU_NUM=0
AVAILABLE_GPU_INDEX=()
for i in $(seq $GPU_NUM); do
  if [ $(echo "${GPU_MEM_USAGE[$i-1]} < 5" | bc) -eq 1 ]; then
    AVAILABLE_GPU_NUM=$((AVAILABLE_GPU_NUM+1))
    AVAILABLE_GPU_INDEX+=($i)
  fi
done
echo "AVAILABLE_GPU_NUM: $AVAILABLE_GPU_NUM"
echo "AVAILABLE_GPU_INDEX: ${AVAILABLE_GPU_INDEX[@]}"


