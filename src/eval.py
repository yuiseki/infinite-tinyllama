import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from time import perf_counter
import sys
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
import yaml
import re

# 指定されたファイルパスからyamlファイルを読み込む
def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data

# 実行時の1番目の引数をload_yamlに渡す
filepath = sys.argv[1]
train_config = load_yaml(filepath)


#
# Migrate
#
model_id = train_config['base_model_id']
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    load_in_8bit=False,
    device_map="auto",
    trust_remote_code=True
)
# Load the tokenizer for the specified model.
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set the padding token to be the same as the end of sentence token.
tokenizer.pad_token = tokenizer.eos_token

# model_path = train_config['output_base_dir']/train_config['model_name']/checkpoint-train_config['train_max_steps']
model_path = os.path.join(train_config['output_base_dir'], train_config['model_name'], f"checkpoint-{train_config['train_max_steps']}")
peft_model = PeftModel.from_pretrained(model, model_path, from_transformers=True, device_map="auto")

model = peft_model.merge_and_unload()


#
# Inference
#

def formatted_prompt(question)-> str:
    template = f"""
    <|im_start|>user
    {question}
    <|im_end|>
    <|im_start|>assistant
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template

def formatted_prompt_with_context(hint, question, context)-> str:
    template = f"""\
    <|im_start|>user
    {hint}
    context:{context}
    question:{question}
    <|im_end|>
    <|im_start|>assistant
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template

def generate_response(user_input):
    prompt = formatted_prompt(user_input)
    generation_config = GenerationConfig(
        penalty_alpha=0.6,
        top_k=5,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.2,
        max_new_tokens=train_config['inference_max_new_tokens'],
        forced_eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, generation_config=generation_config)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return res

def generate_response_with_context(hint, question, context):
    prompt = formatted_prompt_with_context(hint, question, context)
    inputs = tokenizer([prompt], return_tensors="pt")
    generation_config = GenerationConfig(
        penalty_alpha=0.6,
        top_k=5,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.2,
        max_new_tokens=train_config['inference_max_new_tokens'],
        forced_eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, generation_config=generation_config)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return res

def extract_response(output):
    return re.search(r'<\|im_start\|>assistant\n(.+)<\|im_end\|>', output, re.DOTALL).group(1)

if "evaluation_prompts_with_context" in train_config:
    for prompt in train_config['evaluation_prompts_with_context']:
        hint = train_config['dataset_context_hint']
        start_time = perf_counter()
        print(prompt)
        res = generate_response_with_context(hint, prompt['prompt'], prompt['context'])
        print(extract_response(res))
        output_time = perf_counter() - start_time
        print(f"Time taken for inference: {round(output_time,2)} seconds\n\n")
else:
    for prompt in train_config['evaluation_prompts']:
        start_time = perf_counter()
        print(prompt)
        res = generate_response(prompt)
        print(extract_response(res))
        output_time = perf_counter() - start_time
        print(f"Time taken for inference: {round(output_time,2)} seconds\n\n")
