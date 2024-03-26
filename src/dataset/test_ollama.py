import ollama
import time
import sys

text_to_translate = sys.argv[1]

prompt = f"""\
以下の英語を日本語に翻訳してください。翻訳結果のみを簡潔に出力してください。記号は全角にしてください。
{text_to_translate}
"""

models = [
    "gemma:2b-instruct",
    "gemma:7b-instruct",
    "qwen:4b-text",
    "qwen:7b-text",
    "xwinlm:7b",
    "elyza-llama2:7b-instruct",
    "elyza-codellama:7b-instruct",
    "rakutenai:7b-instruct",
    "rinna-youri:7b-instruct",
    "phi:2.7b",
    # "zephyr:7b",
    "yi:6b",
    "stablelm-zephyr:3b",
    "stablelm2:1.6b-zephyr",
]

results = []

for model in models:
    time_start = time.perf_counter()
    print(f"Model: {model}...")
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
    res = response["message"]["content"]
    # 最初の行だけを取り出す
    res = res.split("\n")[0]
    # 先頭の空白を削除
    res = res.lstrip()
    print(res)
    time_end = time.perf_counter()
    time_passed = time_end - time_start
    print(f"Time: {time_passed:.2f}s")
    result = {
        "model": model,
        "response": res,
        "time": time_passed,
    }
    results.append(result)

print("")
print("aggregating...")

# results.responseの内容は確率的にバラバラになっている
# results.responseに同一の結果が出力されていたらそれが正解の可能性が高い
# modelsの数に対して同じ出力がどれだけあるかというmodel-responseをkey-valueで出力する
# type annotationをつける
response_model_num: dict = {}

for result in results:
    if result["response"] in response_model_num:
        response_model_num[result["response"]] += 1
    else:
        response_model_num[result["response"]] = 1

# sort by value
response_model_num = dict(sorted(response_model_num.items(), key=lambda x: x[1], reverse=True))

print("")
print("=========================================")
print("response_model_num")
print(response_model_num)
