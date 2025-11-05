import google.generativeai as genai

# Configure with your Gemini API key
genai.configure(api_key="AIzaSyDv5y1BX5ypCDutMrJF0OvWxtUyA3QFnQA")

# List available models
models = genai.list_models()

print("\n🔍 Available Gemini Models:\n")
for m in models:
    print(f"- {m.name} | Supported methods: {m.supported_generation_methods}")
