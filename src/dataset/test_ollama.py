import ollama
import time

prompt = """\
あなたは優秀な翻訳者です。
以下の英語テキストを日本語に翻訳してください。
翻訳結果のみを出力してください。

Hello, world!
"""

models = [
    "llama2:7b-chat",
    "gemma:7b-instruct",
    "mistral:7b-instruct",
    "elyza:codellama-7b-instruct",
    "qwen:4b-chat",
    "qwen:7b-chat",
    "tinyllama:1.1b-chat",
    "karasu:1b",
]

for model in models:
    time_start = time.perf_counter()
    print("")
    print("=========================================")
    print(f"Model: {model}")
    response = ollama.chat(
        model=model,
        options={
            "temperature": 0.0,
        },
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    print(response["message"]["content"])
    time_end = time.perf_counter()
    print("")
    print(f"Time: {time_end - time_start}")
    print("=========================================")
