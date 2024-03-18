import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import numpy as np

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# https://note.com/kan_hatakeyama/n/nb5625d6411a8

def perplexity(model, tokenizer, text) -> torch.Tensor:
    tokenized_input = tokenizer.encode(
          text,
          add_special_tokens = False,
          return_tensors = "pt"
    ).to(model.device)
    with torch.inference_mode():
        output = model(tokenized_input, labels = tokenized_input)
    ppl = torch.exp(output.loss)
    return ppl.item()

class MoE:
    def __init__(self):
        self.models = []
        self.coef = []

    def set_coefs(self, coef):
        self.coef = coef

    def append_ELM(self, model, tokenizer):
        self.models.append((model, tokenizer))
        self.coef.append(1)

    def calc_perplexity(self, text):
        ppl_list = []
        for model, tokenizer in self.models:
            ppl_list.append(perplexity(model, tokenizer, text))

        return ppl_list

    def ask(self, text, verbose = True):
        ppl_array = np.array(self.calc_perplexity(text))
        ppl_array = ppl_array * np.array(self.coef)
        best_model_id = np.where(ppl_array == min(ppl_array))[0][0]
        if verbose:
            print("perplexity list")
            for i, ppl in enumerate(ppl_array):
                print(i, ppl)
            print(f"model id {best_model_id} is used")

        tokenizer = self.models[best_model_id][1]
        generation_config = GenerationConfig(
            penalty_alpha=0.6,
            top_k=5,
            do_sample=True,
            temperature=0.1,
            repetition_penalty=1.2,
            max_new_tokens=100,
            forced_eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        inputs = tokenizer(text, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs, generation_config=generation_config)
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return res

moe = MoE()

model_path_list =[
    "",
    "output/tinyllama-color-coder-v1/checkpoint-200",
]

for model_path in model_path_list:
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype = torch.float16,
        load_in_8bit = False,
        device_map = "auto",
        trust_remote_code = True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if model_path == "":
        moe.append_ELM(model, tokenizer)
        continue

    peft_model = PeftModel.from_pretrained(
        model,
        model_path,
        from_transformers = True,
        device_map = "auto"
    )
    model = peft_model.merge_and_unload()
    moe.append_ELM(model, tokenizer)

moe.set_coefs([1, 1])

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

text_list = [
    "Who is the Secretary General of the United Nations?",
    "Pure Black: A shade that completely absorbs light and does not reflect any colors. It is the darkest possible shade.",
]

for text in text_list:
    print("-----")
    prompt = formatted_prompt(text)
    response = moe.ask(text)
    print(response)
