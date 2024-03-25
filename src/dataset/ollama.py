import ollama

response = ollama.chat(model='gemma:7b-instruct', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
