# https://www.analyticsvidhya.com/blog/2024/02/sql-generation-in-text2sql-with-tinyllamas-llm-fine-tuning/

import torch

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from transformers import TrainingArguments
from peft import AutoPeftModelForCausalLM, PeftModel, LoraConfig
from trl import SFTTrainer


# Define the model to fine-tune
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Define the dataset for fine-tuning
dataset_id = "b-mc2/sql-create-context"

data = load_dataset(dataset_id, split="train")
df = data.to_pandas()

def chat_template_for_training(context, answer, question):
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


# Apply the chat_template_for_training function to each row in the 
# dataframe and store the result in a new "text" column.
df["text"] = df.apply(lambda x: chat_template_for_training(x["context"], x["answer"], x["question"]), axis=1)

# Convert the dataframe back to a Dataset object.
formatted_data = Dataset.from_pandas(df)

print(df['text'][1])



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
    output_dir="./output/tinyllama-sql-v1",
    # Set the per-device training batch size.
    per_device_train_batch_size=8,
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
    max_steps=250,
    # Enable fp16 training.
    fp16=True,
)



# Initialize the SFTTrainer.
trainer = SFTTrainer(
    # Set the model to be trained.
    model=model,
    # Set the training dataset.
    train_dataset=formatted_data,
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

trainer.train()





# Load the pre-trained model.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, 
    load_in_8bit=False, 
    device_map="auto",  
    trust_remote_code=True 
)

# Load the PEFT model from a checkpoint.
model_path = "./output/tinyllama-sql-v1/checkpoint-250"
peft_model = PeftModel.from_pretrained(model, model_path, from_transformers=True, device_map="auto")

# Wrap the model with the PEFT model.
model = peft_model.merge_and_unload()

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
output = model.generate(**inputs, max_new_tokens=512)

# Decode the output.
text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated SQL query.
print(text)
