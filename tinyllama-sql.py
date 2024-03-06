# https://www.analyticsvidhya.com/blog/2024/02/sql-generation-in-text2sql-with-tinyllamas-llm-fine-tuning/

import torch

from datasets import load_dataset, Dataset

from peft import AutoPeftModelForCausalLM, PeftModel, LoraConfig

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, GenerationConfig

from trl import SFTTrainer

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


# Define the model to fine-tune
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Define the dataset for fine-tuning
dataset_id = "b-mc2/sql-create-context"

# Define the name of the output model
output_model = "tinyllama-sql-v1"

# Define the path to the output model
output_path = "./output/tinyllama-sql-v1"

# Define the path to the pre-trained model
model_path = "./output/tinyllama-sql-v1/checkpoint-200"

def template_for_train(context, answer, question):
    template = f"""\
    <|im_start|>user
    Given the context, generate an SQL query for the following question
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

def prepare_train_data(dataset_id):
    data = load_dataset(dataset_id, split="train")
    df = data.to_pandas()
    # Apply the chat_template_for_training function to each row in the 
    # dataframe and store the result in a new "text" column.
    df["text"] = df.apply(lambda x: template_for_train(x["context"], x["answer"], x["question"]), axis=1)
    # Convert the dataframe back to a Dataset object.
    data = Dataset.from_pandas(df)
    data = data.train_test_split(seed=42, test_size=0.2)
    return data

data = prepare_train_data(dataset_id)

# print(data["train"][0]["text"])
# print(data["train"][1]["text"])

#
# Load the model and tokenizer
#
def get_model_and_tokenizer(model_id):
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

model, tokenizer = get_model_and_tokenizer(model_id)

#
# LoRA and PEFT config
#
# Define the PEFT configuration.
peft_config = LoraConfig(
    # Set the rank of the LoRA projection matrix.
    r=8,
    # Set the alpha parameter for the LoRA projection matrix.
    lora_alpha=16,
    # Set the dropout rate for the LoRA projection matrix.
    lora_dropout=0.05,
    # Set the bias term to "none".
    bias="none",
    # Set the task type to "CAUSAL_LM".
    task_type="CAUSAL_LM"
)

# Define the training arguments.
training_args = TrainingArguments(
    # Set the output directory for the training run.
    output_dir=output_path,
    # Set the per-device training batch size.
    per_device_train_batch_size=6,
    # Set the number of gradient accumulation steps.
    gradient_accumulation_steps=2,
    # Set the optimizer to use.
    optim="paged_adamw_32bit",
    # Set the learning rate.
    learning_rate=2e-4,
    # Set the learning rate scheduler type.
    lr_scheduler_type="cosine",
    # Set the save strategy.
    save_strategy="epoch",
    # Set the logging steps.
    logging_steps=10,
    # Set the number of training epochs.
    num_train_epochs=2,
    # Set the maximum number of training steps.
    max_steps=1000,
    # Enable fp16 training.
    fp16=True,
)

# Initialize the SFTTrainer.
trainer = SFTTrainer(
    # Set the model to be trained.
    model=model,
    # Set the training dataset.
    train_dataset=data["train"],
    # Set the evaluation dataset.
    eval_dataset=data["test"],
    # Set the PEFT configuration.
    peft_config=peft_config,
    # Set the name of the text field in the dataset.
    dataset_text_field="text",
    # Set the training arguments.
    args=training_args,
    # Set the tokenizer.
    tokenizer=tokenizer,
    # Disable packing.
    packing=False,
    # Set the maximum sequence length.
    max_seq_length=1024
)

#
# Execute train
#
trainer.train()
#
#
#

#
# Migrate
#
# Load the pre-trained model.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, 
    load_in_8bit=False, 
    device_map="auto",  
    trust_remote_code=True 
)

# Load the PEFT model from a checkpoint.
peft_model = PeftModel.from_pretrained(model, model_path, from_transformers=True, device_map="auto")

# Wrap the model with the PEFT model.
model = peft_model.merge_and_unload()


#
# Inference
#
def chat_template(question, context):
    template = f"""\
    <|im_start|>user
    Given the context, generate an SQL query for the following question
    context:{context}
    question:{question}
    <|im_end|>
    <|im_start|>assistant 
    """
    # Remove any leading whitespace characters from each line in the template.
    template = "\n".join([line.lstrip() for line in template.splitlines()])
    return template

# Prepare the Prompt.
question = "How many heads of the departments are older than 56 ?"
context = "CREATE TABLE head (age INTEGER)"
prompt = chat_template(question,context)

# Encode the prompt.
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

# Generate the output.
generation_config = GenerationConfig(
    penalty_alpha=0.6,
    top_k=5,
    do_sample=True,
    temperature=0.1,
    repetition_penalty=1.2,
    max_new_tokens=32,
    pad_token_id=tokenizer.eos_token_id
)
output = model.generate(**inputs, generation_config=generation_config)

# Decode the output.
text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated SQL query.
print(text)
