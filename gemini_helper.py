from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def explain(prediction, probability, recommendations, values):
    prompt = f"""
You are an AI assistant for a paper manufacturing plant.

Prediction:
{prediction}

Risk Score:
{probability:.2%}

Sensor Values:
{values}

Recommended Actions:
{recommendations}

Explain:

• Root cause

• Why these variables matter

• Why the recommendations reduce risk

Keep the explanation under 120 words.
"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
    )
    return response.text