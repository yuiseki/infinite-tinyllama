# https://www.kaggle.com/code/ssarkar445/huggingface-tinyllama-finetune-peft

import torch

from datasets import load_dataset, Dataset

from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GenerationConfig

from trl import SFTTrainer

from time import perf_counter

# model_id="TinyLlama-1.1B-Chat-v0.3"
model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Define the dataset for fine-tuning
dataset="burkelibbey/colors"

# Define the name of the output model
output_model="tinyllama-colorist-v1"

# Define the path to the output model
output_path="./output/tinyllama-colorist-v1"

# Define the path to the pre-trained model
model_path = "./output/tinyllama-colorist-v1/checkpoint-200"


#
# Prepare train data
#
def formatted_train(input,response)->str:
    return f"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"

def prepare_train_data(data_id):
    data = load_dataset(data_id, split="train")
    data_df = data.to_pandas()
    data_df["text"] = data_df[["description", "color"]].apply(lambda x: formatted_train(x["description"], x["color"]), axis=1)
    data = Dataset.from_pandas(data_df)
    data = data.train_test_split(seed=42, test_size=0.2)
    return data

data = prepare_train_data(dataset)

# print(data["train"][0]["text"])

#
# Load model
#
def get_model_and_tokenizer(mode_id):
    tokenizer = AutoTokenizer.from_pretrained(mode_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        # 謎のパラメーター
        load_in_4bit=True,
        # 謎のパラメーター
        bnb_4bit_quant_type="nf4",
        # 謎のパラメーター
        bnb_4bit_compute_dtype="float16",
        # 謎のパラメーター
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        mode_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_id)

#
# LoRA and PEFT config
#
peft_config = LoraConfig(
        # 謎のパラメーター
        r=8,
        # 謎のパラメーター
        lora_alpha=16,
        # 謎のパラメーター
        lora_dropout=0.05,
        # 謎のパラメーター
        bias="none",
        # 謎のパラメーター
        task_type="CAUSAL_LM"
    )

training_arguments = TrainingArguments(
        output_dir=output_path,
        # 謎のパラメーター
        per_device_train_batch_size=16,
        # 謎のパラメーター
        gradient_accumulation_steps=8,
        # 謎のパラメーター
        optim="paged_adamw_32bit",
        # 謎のパラメーター
        learning_rate=2e-4,
        # 謎のパラメーター
        lr_scheduler_type="cosine",
        # 謎のパラメーター
        save_strategy="epoch",
        # 謎のパラメーター
        logging_steps=20,
        # 謎のパラメーター
        num_train_epochs=2,
        # 謎のパラメーター
        max_steps=200,
        # 謎のパラメーター
        fp16=True,
        push_to_hub=False
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
# trainer.train()
#
#
#


#
# Migrate
#
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.float16,
                                             load_in_8bit=False,
                                             device_map="auto",
                                             trust_remote_code=True)

peft_model = PeftModel.from_pretrained(model, model_path, from_transformers=True, device_map="auto")

model = peft_model.merge_and_unload()


#
# Inference
#
def formatted_prompt(question)-> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"

def generate_response(user_input):
  prompt = formatted_prompt(user_input)
  inputs = tokenizer([prompt], return_tensors="pt")
  generation_config = GenerationConfig(
      penalty_alpha=0.6,
      top_k=5,
      do_sample=True,
      temperature=0.1,
      repetition_penalty=1.2,
      max_new_tokens=12,
      pad_token_id=tokenizer.eos_token_id
  )
  start_time = perf_counter()
  inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
  outputs = model.generate(**inputs, generation_config=generation_config)
  res = tokenizer.decode(outputs[0], skip_special_tokens=True)
  output_time = perf_counter() - start_time
  print(f"\nTime taken for inference: {round(output_time,2)} seconds\n")
  return res

def print_color_space(hex_color):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = hex_to_rgb(hex_color)
    print(f'{hex_color}: \033[48;2;{r};{g};{b}m           \033[0m')

res1 = generate_response(user_input='Pure Black: A shade that completely absorbs light and does not reflect any colors. It is the darkest possible shade.')
print(res1)

res2 = generate_response(user_input='give me a pure brown color')
print(res2)

res3 = generate_response(user_input='Light Orange color')
print(res3)
