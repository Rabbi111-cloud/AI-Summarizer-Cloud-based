import os
import json
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
app = FastAPI(
    title="Cloud AI Summarizer & Sentiment Analyzer",
    description="POST endpoints to summarize text & analyze sentiment via OpenRouter.",
    version="1.1.0"
)

class TextRequest(BaseModel):
    text: str

# -----------------------------
# OpenRouter Caller
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
        r = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=40)
        data = r.json()

        if "choices" not in data:
            return f"Error: No choices. Raw response: {data}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Cloud AI Summarizer & Sentiment Analyzer is running!"}

@app.post("/summarize")
def summarize(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(400, "No text provided.")
    summary = call_openrouter(
        f"Summarize the following text in 3–6 sentences:\n\n{request.text}"
    )
    return {"summary": summary}

@app.post("/sentiment")
def sentiment(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(400, "No text provided.")

    raw = call_openrouter(
        f"""
        Analyze the sentiment of the text below. 
        Return JSON ONLY in this format:
        {{
            "sentiment": "Positive | Neutral | Negative",
            "confidence": 0.0–1.0,
            "explanation": "One-sentence explanation"
        }}

        Text:
        {request.text}
        """
    )

    # Convert returned JSON-string to python dict
    try:
        sentiment_json = json.loads(raw)
    except:
        sentiment_json = {"sentiment": "Unknown", "raw": raw}

    return {"sentiment": sentiment_json}

@app.post("/analyze")
def analyze(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(400, "No text provided.")

    summary = call_openrouter(
        f"Summarize the following text in 3–6 sentences:\n\n{request.text}"
    )

    raw_sentiment = call_openrouter(
        f"""
        Analyze the sentiment of this text and return JSON (NO explanation outside JSON):

        {{
            "sentiment": "Positive | Neutral | Negative",
            "confidence": 0.0–1.0,
            "explanation": "One-sentence explanation"
        }}

        Text:
        {request.text}
        """
    )

    try:
        sentiment_json = json.loads(raw_sentiment)
    except:
        sentiment_json = {"sentiment": "Unknown", "raw": raw_sentiment}

    return {
        "summary": summary,
        "sentiment": sentiment_json
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
