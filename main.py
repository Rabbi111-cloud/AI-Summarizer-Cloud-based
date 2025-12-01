import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------
# Environment Variables
# -----------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY is not set. API calls will fail.")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Cloud AI Summarizer & Sentiment Analyzer")

# -----------------------------
# Request model
# -----------------------------
class TextRequest(BaseModel):
    text: str

# -----------------------------
# Generic OpenRouter Call
# -----------------------------
def call_openrouter(prompt: str, model: str = MODEL) -> str:
    if not OPENROUTER_API_KEY:
        return "Error: API key not set."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=30)
        data = response.json()
        choices = data.get("choices")
        if choices and isinstance(choices, list):
            return choices[0].get("message", {}).get("content", "")
        else:
            return f"Error: No choices returned. Raw response: {data}"
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# Routes
# -----------------------------
@app.post("/summarize")
def summarize(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided.")
    result = call_openrouter(
        f"Summarize the following text in 3-6 sentences:\n\n{request.text}"
    )
    return {"summary": result}


@app.post("/sentiment")
def sentiment(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided.")
    result = call_openrouter(
        f"Analyze the sentiment of the following text. "
        "Return JSON with keys: sentiment (Positive, Neutral, Negative), "
        "confidence (0-1), explanation (one sentence).\n\nText:\n{request.text}"
    )
    return {"sentiment": result}


@app.post("/analyze")
def analyze(request: TextRequest):
    """
    Combined endpoint: returns both summary and sentiment.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided.")

    summary = call_openrouter(
        f"Summarize the following text in 3-6 sentences:\n\n{request.text}"
    )
    sentiment_result = call_openrouter(
        f"Analyze the sentiment of the following text. "
        "Return JSON with keys: sentiment (Positive, Neutral, Negative), "
        "confidence (0-1), explanation (one sentence).\n\nText:\n{request.text}"
    )
    return {"summary": summary, "sentiment": sentiment_result}
