import ollama
import time

prompt = """\
あなたは優秀な翻訳者です。
以下の英語テキストを日本語に翻訳してください。
翻訳結果のみを出力してください。

Hello, world!
"""

models = [
    "gemma:2b-instruct",
    "gemma:7b-instruct",
    "elyza-llama2:7b-instruct",
    "elyza-codellama:7b-instruct",
    "qwen:4b-chat",
    "qwen:7b-chat",
    "zephyr:7b",
]

results = []

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
    res = response["message"]["content"]
    # 最初の行だけを取り出す
    res = res.split("\n")[0]
    # 先頭の空白を削除
    res = res.lstrip()
    print(res)
    time_end = time.perf_counter()
    print("")
    print(f"Time: {time_end - time_start}")
    print("=========================================")
    result = {
        "model": model,
        "response": res,
        "time": time_end - time_start,
    }
    results.append(result)

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

print("")
print("=========================================")
print("response_model_num")
print(response_model_num)
