import os
import sys

import torch
import yaml
from accelerate import PartialState
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

import wandb

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
# Template
#
def simple_template_for_pretrain(input) -> str:
    # inputから、2つ以上連続する改行を除去する
    input = "\n".join([line for line in input.splitlines() if line.strip() != ""])
    template = input
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template


def simple_template_for_train(input, output) -> str:
    template = f"""\
    <|im_start|>user
    {input}
    <|im_end|>
    <|im_start|>assistant
    {output}
    <|im_end|>\
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template


def hint_template_for_train(hint, question, answer):
    template = f"""\
    <|im_start|>user
    {hint}
    {question}
    <|im_end|>
    <|im_start|>assistant
    {answer}
    <|im_end|>\
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template


def context_template_for_train(context, question, answer):
    template = f"""\
    <|im_start|>user
    {question}
    {context}
    <|im_end|>
    <|im_start|>assistant
    {answer}
    <|im_end|>\
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template


def context_hint_template_for_train(hint, context, question, answer):
    template = f"""\
    <|im_start|>user
    {hint}
    context:
    {context}
    question:
    {question}
    <|im_end|>
    <|im_start|>assistant
    {answer}
    <|im_end|>\
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template


#
# Prepare train data
#
def prepare_train_data(dataset_id):
    if "dataset_load_config" in train_config:
        dataset_load_config = train_config["dataset_load_config"]
        data = load_dataset(dataset_id, dataset_load_config, split="train", num_proc=32)
        if dataset_load_config == "20231101.ja" or dataset_load_config == "20231101.vi" or dataset_load_config == "20231101.es" or dataset_load_config == "20231101.it":
            data = data.filter(lambda item, idx: idx % 3 == 0, with_indices=True)
        if dataset_load_config == "20231101.de" or dataset_load_config == "20231101.fr":
            data = data.filter(lambda item, idx: idx % 5 == 0, with_indices=True)
    else:
        data = load_dataset(dataset_id, split="train", num_proc=32)

    data_df = data.to_pandas()

    if "dataset_filter_field_name" in train_config:
        data_df = data_df[data_df[train_config["dataset_filter_field_name"]] == train_config["dataset_filter_field_value"]]

    input_field_name = train_config["dataset_input_field_name"]
    if "dataset_output_field_name" not in train_config:
        data_df["text"] = data_df[input_field_name].apply(lambda x: simple_template_for_pretrain(x))
    else:
        output_field_name = train_config["dataset_output_field_name"]
        if "dataset_output_field_values_to_texts" in train_config:
            output_field_values_to_texts = train_config["dataset_output_field_values_to_texts"]
            data_df[output_field_name] = data_df[output_field_name].apply(lambda x: output_field_values_to_texts.get(x, x))
        if "dataset_context_field_name" in train_config:
            context_field_name = train_config["dataset_context_field_name"]
            if "dataset_context_hint" not in train_config:
                data_df["text"] = data_df[[context_field_name, input_field_name, output_field_name]].apply(
                    lambda x: context_template_for_train(x[context_field_name], x[input_field_name], x[output_field_name]),
                    axis=1,
                )
            else:
                context_hint = train_config["dataset_context_hint"]
                data_df["text"] = data_df[[context_field_name, input_field_name, output_field_name]].apply(
                    lambda x: context_hint_template_for_train(
                        context_hint,
                        x[context_field_name],
                        x[input_field_name],
                        x[output_field_name],
                    ),
                    axis=1,
                )
        elif "dataset_input_hint" in train_config:
            input_hint = train_config["dataset_input_hint"]
            data_df["text"] = data_df[[input_field_name, output_field_name]].apply(
                lambda x: hint_template_for_train(input_hint, x[input_field_name], x[output_field_name]),
                axis=1,
            )
        else:
            data_df["text"] = data_df[[input_field_name, output_field_name]].apply(
                lambda x: simple_template_for_train(x[input_field_name], x[output_field_name]),
                axis=1,
            )
    # keep only text field
    data = data_df[["text"]]
    data = Dataset.from_pandas(data_df)
    data = data.train_test_split(seed=42, test_size=0.2)
    print(len(data["train"]))
    return data


dataset_id = train_config["dataset_id"]
data = prepare_train_data(dataset_id)


#
# Load the model and tokenizer
#
def load_model_and_tokenizer(model_id):
    # Load the tokenizer for the specified model.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # NOTE: これやるならmodel.resize_token_embeddingsが必要
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # NOTE: tokenizer.add_special_tokensやるならこれは不要
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Define the quantization configuration for memory-efficient training.
    bnb_config = BitsAndBytesConfig(
        # Load the model weights in 4-bit quantized format.
        load_in_4bit=True,
        # Specify whether to use double quantization for 4-bit quantization.
        bnb_4bit_use_double_quant=True,
        # Specify the quantization type to use for 4-bit quantization.
        bnb_4bit_quant_type="nf4",
        # Specify the data type to use for computations during training.
        bnb_4bit_compute_dtype=torch.float16,
    )
    # Load the model from the specified model ID and apply the quantization configuration.

    model = AutoModelForCausalLM.from_pretrained(
        # Base model id
        model_id,
        # BitsAndBytes configuration
        quantization_config=bnb_config,
        # Set torch dtype
        torch_dtype=torch.float16,
        # Trust remote code
        trust_remote_code=True,
        # Set low cpu mem usage
        low_cpu_mem_usage=True,
        # Set device map to auto
        # device_map="auto",
        device_map={"": PartialState().process_index},
        # Set the attention impl
        # if model_id is llm-jp/llm-jp-13b-v1.0, dont use flash_attention
        # if other models, use flash_attention_2
        attn_implementation=None if "llm-jp-13b" in model_id else "flash_attention_2",
    )
    # Disable cache to improve training speed.
    model.config.use_cache = False
    # Set the temperature for pretraining to 1.
    model.config.pretraining_tp = 1
    # NOTE: tokenizer.add_special_tokensやるならこれが必要
    # model.resize_token_embeddings(len(tokenizer))
    print(model.hf_device_map)
    return model, tokenizer


model_id = train_config["base_model_id"]

output_dir = os.path.join(train_config["output_base_dir"], train_config["model_name"])
merged_model_path = os.path.join(output_dir, "pretrained")

# model_pathが既にある場合は終了
if os.path.exists(merged_model_path):
    print(f"{merged_model_path} already exists.")
    sys.exit(1)


model, tokenizer = load_model_and_tokenizer(model_id)


os.environ["WANDB_PROJECT"] = "infinite-tinyllama"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "all"

#
# Define LoRA and PEFT config
#

peft_config = LoraConfig(
    r=int(train_config["lora_r"]),
    lora_alpha=int(train_config["lora_alpha"]),
    lora_dropout=float(train_config["lora_dropout"]),
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    report_to="wandb",
    per_device_train_batch_size=int(train_config["train_per_device_train_batch_size"]),
    gradient_accumulation_steps=int(train_config["train_gradient_accumulation_steps"]),
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=int(train_config["train_num_train_epochs"]),
    fp16=True,
    run_name=train_config["model_name"],
)

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_arguments,
    tokenizer=tokenizer,
    packing=False,
    max_seq_length=1024,
)

#
# Execute train
#
trainer.train()


#
# Save the model
#
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained(merged_model_path)
# tokenizer.save_pretrained(merged_model_path)

if PartialState().is_last_process:
    wandb.finish()
