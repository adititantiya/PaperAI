import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

print("API Key:", os.getenv("GEMINI_API_KEY")[:10] + "...")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models = genai.list_models()

for model in models:
    print(model.name)