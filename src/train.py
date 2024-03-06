import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from trl import SFTTrainer
from time import perf_counter
import sys
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
import yaml


# 指定されたファイルパスからyamlファイルを読み込む
def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data

# 実行時の1番目の引数をload_yamlに渡す
filepath = sys.argv[1]
train_config = load_yaml(filepath)

#
# Prepare train data
#
def prepare_train_data(dataset_id):
    input_field_name = train_config['dataset_input_field_name']
    output_field_name = train_config['dataset_output_field_name']
    def simple_template_for_train(input, output)->str:
        template = f"""\
        <|im_start|>user
        {input}
        <|im_end|>
        <|im_start|>assistant
        {output}
        <|im_end|>
        """
        # Remove any leading whitespace characters from each line in the template.
        template = "\n".join([line.lstrip() for line in template.splitlines()])
        return template

    def context_template_for_train(hint, context, question, answer):
        template = f"""\
        <|im_start|>user
        {hint}
        context:{context}
        question:{question}
        <|im_end|>
        <|im_start|>assistant
        {answer}
        <|im_end|>
        """
        # Remove any leading whitespace characters from each line in the template.
        template = "\n".join([line.lstrip() for line in template.splitlines()])
        return template

    data = load_dataset(dataset_id, split="train")
    data_df = data.to_pandas()

    if "dataset_input_context_field_name" in train_config:
        context_field_name = train_config['dataset_input_context_field_name']
        context_hint = train_config['dataset_input_context_hint']
        data_df["text"] = data_df[[context_field_name, input_field_name, output_field_name]].apply(lambda x: context_template_for_train(context_hint, x[context_field_name], x[input_field_name], x[output_field_name]), axis=1)
    else:
        data_df["text"] = data_df[[input_field_name, output_field_name]].apply(lambda x: simple_template_for_train(x[input_field_name], x[output_field_name]), axis=1)

    data = Dataset.from_pandas(data_df)
    data = data.train_test_split(seed=42, test_size=0.2)
    return data

dataset_id = train_config['dataset_id']
data = prepare_train_data(dataset_id)

#
# Load the model and tokenizer
#
def load_model_and_tokenizer(model_id):
    # Load the tokenizer for the specified model.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Set the padding token to be the same as the end of sentence token.
    tokenizer.pad_token = tokenizer.eos_token

    # Define the quantization configuration for memory-efficient training.
    bnb_config = BitsAndBytesConfig(
        # Load the model weights in 4-bit quantized format.
        load_in_4bit=True,
        # Specify the quantization type to use for 4-bit quantization.
        bnb_4bit_quant_type="nf4",
        # Specify the data type to use for computations during training.
        bnb_4bit_compute_dtype="float16",
        # Specify whether to use double quantization for 4-bit quantization.
        bnb_4bit_use_double_quant=True
    )
    # Load the model from the specified model ID and apply the quantization configuration.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    # Disable cache to improve training speed.
    model.config.use_cache = False
    # Set the temperature for pretraining to 1.
    model.config.pretraining_tp = 1 
    return model, tokenizer

model_id = train_config['base_model_id']
model, tokenizer = load_model_and_tokenizer(model_id)


#
# Define LoRA and PEFT config
#

peft_config = LoraConfig(
        r=int(train_config['lora_r']),
        lora_alpha=int(train_config['lora_alpha']),
        lora_dropout=float(train_config['lora_dropout']),
        bias="none",
        task_type="CAUSAL_LM"
    )
output_dir = os.path.join(train_config['output_base_dir'], train_config['model_name'])
training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(train_config['train_per_device_train_batch_size']),
        gradient_accumulation_steps=int(train_config['train_gradient_accumulation_steps']),
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=int(train_config['train_num_train_epochs']),
        max_steps=int(train_config['train_max_steps']),
        fp16=True
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
        max_seq_length=1024
    )


#
# Execute train
#
trainer.train()
#
#
#
