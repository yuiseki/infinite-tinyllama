# Autonomous Agent that train LLMs written by shell script
#
# Description
# This script is the main script for the autonomous agent. It is responsible for the followings
# 1. Monitoring the state of the computational resources
# 2. Making decisions based on the state of the computational resources
#   2.1. Keep waiting - Continue to wait until computing resources are available
#   2.2. Search task - Search task on the available computing resources
#   2.3. Generate recipes - Generate recipes for the new LLM based on the state of the computational resources and the found task


eval "$(/home/yuiseki/miniconda3/bin/conda shell.bash hook)"
export PATH="/home/yuiseki/miniconda3/bin:$PATH"
conda activate peft

# $1 = directory of recipes
RECIPE_DIR=$1
# $2 = MAX_ALLOWED_GPU_NUM
MAX_ALLOWED_GPU_NUM=$2

# Set CUDA_VISIBLE_DEVICES as the first available GPUs based on MAX_ALLOWED_GPU_NUM
CUDA_VISIBLE_DEVICES=""
for i in $(seq $MAX_ALLOWED_GPU_NUM); do
  CUDA_VISIBLE_DEVICES+="${AVAILABLE_GPU_INDEX[$i-1]}"
  if [ $i -lt $MAX_ALLOWED_GPU_NUM ]; then
    CUDA_VISIBLE_DEVICES+=","
  fi
done
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# echo all recipes path
echo "RECIPE_DIR: $RECIPE_DIR"
RECIPE_LIST=$(ls $RECIPE_DIR)
AVAILABLE_RECIPES=()
for RECIPE in $RECIPE_LIST; do
  RECIPE_CLAIM_GPU_NUM=$(cat $1$RECIPE | yq ".train_claim_gpu_num")
  # If RECIPE_CLAIM_GPU_NUM is greater than MAX_ALLOWED_GPU_NUM, skip
  if [ $RECIPE_CLAIM_GPU_NUM -gt $MAX_ALLOWED_GPU_NUM ]; then
    continue
  fi
  if [ $RECIPE_CLAIM_GPU_NUM -le $AVAILABLE_GPU_NUM ]; then
    AVAILABLE_RECIPES+=($RECIPE)
  fi
done

echo "AVAILABLE_RECIPES: ${AVAILABLE_RECIPES[@]}"

for RECIPE in $AVAILABLE_RECIPES; do
  TARGET_MODEL_NAME=$(cat $1$RECIPE | yq ".model_name")
  echo "TARGET_MODEL_NAME: $TARGET_MODEL_NAME"
  # If recipe already trained, skip
  # If recipe already trained, directory like output/TARGET_MODEL_NAME are exists
  # If exists, skip
  if [ -d "output/$TARGET_MODEL_NAME" ]; then
    echo "Skip $1$RECIPE"
    continue
  fi
  # If recipe not trained, train the recipe
  echo "Train $1$RECIPE"
  python src/train.py $1$RECIPE
done
