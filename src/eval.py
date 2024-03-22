import os
import re
import sys

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


# 指定されたファイルパスからyamlファイルを読み込む
def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


# 実行時の1番目の引数をload_yamlに渡す
filepath = sys.argv[1]
train_config = load_yaml(filepath)


#
# Utils
#
def formatted_prompt(question) -> str:
    template = f"""\
    <|im_start|>user
    {question}
    <|im_end|>
    <|im_start|>assistant
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template


def formatted_prompt_with_hint(hint, question) -> str:
    template = f"""\
    <|im_start|>user
    {hint}
    {question}
    <|im_end|>
    <|im_start|>assistant
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template


def formatted_prompt_with_context(question, context) -> str:
    template = f"""\
    <|im_start|>user
    {question}
    {context}
    <|im_end|>
    <|im_start|>assistant
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template


def formatted_prompt_with_hint_and_context(hint, question, context) -> str:
    template = f"""\
    <|im_start|>user
    {hint}
    context:
    {context}
    question:
    {question}
    <|im_end|>
    <|im_start|>assistant
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template


def generate_response(target_model, prompt):
    generation_config = GenerationConfig(
        penalty_alpha=0.6,
        top_k=1,
        do_sample=True,
        temperature=0.001,
        repetition_penalty=1.2,
        max_new_tokens=train_config["inference_max_new_tokens"],
        forced_eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = target_model.generate(**inputs, generation_config=generation_config)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return res


def extract_user_input(prompt):
    return re.search(
        r"<\|im_start\|>user\n(.+)\n<\|im_end\|>", prompt, re.DOTALL
    ).group(1)


def extract_assistant_output(output):
    return re.search(
        r"<\|im_start\|>assistant\n(.+)\n<\|im_end\|>", output, re.DOTALL
    ).group(1)


def extract_assistant_output_robust(output):
    # <|im_start|>assistant はあるけど <|im_end|> がない場合は、<|im_start|>assistant から最後までを返す
    match = re.search(r"<\|im_start\|>assistant\n(.+)", output, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return output


#
# Evaluate
#
def evaluate_model(model, train_config):
    score = 0
    for evaluation in train_config["evaluations"]:
        input = evaluation["prompt"]
        expected_output = evaluation["expected_output"]
        if "context" in evaluation:
            if "dataset_context_hint" not in train_config:
                prompt = formatted_prompt_with_context(input, evaluation["context"])
            else:
                hint = train_config["dataset_context_hint"]
                prompt = formatted_prompt_with_hint_and_context(
                    hint, input, evaluation["context"]
                )
        elif "hint" in evaluation:
            hint = evaluation["hint"]
            prompt = formatted_prompt_with_hint(hint, input)
        else:
            prompt = formatted_prompt(input)
        extracted_input = extract_user_input(prompt)
        print("Prompt:")
        print(extracted_input)
        res = generate_response(model, prompt)
        # print(res)
        try:
            extracted_output = extract_assistant_output(res)
        except AttributeError:
            extracted_output = extract_assistant_output_robust(res)
        print(f"Expected Output:\t{expected_output}")
        print(f"Actual Output:\t\t{extracted_output}")
        print("\n\n")
        if extracted_output == expected_output:
            score += 1
    return score


#
# Original model
#
model_id = train_config["base_model_id"]
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    load_in_8bit=False,
    device_map="auto",
    trust_remote_code=True,
)
# Load the tokenizer for the specified model.
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set the padding token to be the same as the end of sentence token.
tokenizer.pad_token = tokenizer.eos_token

#
# Evaluate base model
#
print("===== ===== ===== Evaluating base model... ===== ===== =====")
base_model_score = evaluate_model(base_model, train_config)

#
# Merge model
#
model_path = os.path.join(
    train_config["output_base_dir"],
    train_config["model_name"],
    f"checkpoint-{train_config['train_max_steps']}",
)
peft_model = PeftModel.from_pretrained(
    base_model, model_path, from_transformers=True, device_map="auto"
)

merged_model = peft_model.merge_and_unload()

#
# Evaluate merged model
#
print("===== ===== ===== Evaluating merged model... ===== ===== =====")
merged_model_score = evaluate_model(merged_model, train_config)

print(f"Base model score: {base_model_score}")
print(f"Merged model score: {merged_model_score}")
