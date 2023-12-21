import litellm
from litellm import completion
litellm.set_verbose = True

response = completion(
    model="ollama/orca-mini",
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    api_base="http://localhost:11434",
    stream=True
)
print(response)
for chunk in response:
    print(chunk['choices'][0]['delta'])
