import ollama

prompt = f"""\
以下のテキストを日本語に翻訳します。翻訳結果を```で囲みます。
"Hello, world!"
```
"""

response = ollama.chat(model='gemma:7b-text', messages=[
  {
    'role': 'user',
    'content': prompt,
  },
])
print(response['message']['content'])
