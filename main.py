 import os
import requests
import json

# -----------------------------
# Environment Variables
# -----------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")  # optional override

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY is not set. API calls will fail.")


# -----------------------------
# Generic OpenRouter Call
# -----------------------------
def call_openrouter(prompt: str, model: str = MODEL) -> str:
    """
    Call OpenRouter API and return the text response.
    """
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
        # Defensive parsing
        choices = data.get("choices")
        if choices and isinstance(choices, list):
            return choices[0].get("message", {}).get("content", "")
        else:
            return f"Error: No choices returned. Raw response: {json.dumps(data)}"
    except Exception as e:
        return f"Error: {str(e)}"


# -----------------------------
# Summarization Function
# -----------------------------
def summarize_text(text: str) -> str:
    if not text:
        return "Error: No text provided for summarization."

    prompt = f"""
You are a concise summarization assistant. Summarize the following text in 3-6 sentences:

{text}
"""
    return call_openrouter(prompt)


# -----------------------------
# Sentiment Analysis Function
# -----------------------------
def analyze_sentiment(text: str) -> str:
    if not text:
        return "Error: No text provided for sentiment analysis."

    prompt = f"""
Analyze the sentiment of the following text and return JSON with keys:
- sentiment: Positive, Neutral, Negative
- confidence: 0-1
- explanation: one sentence explanation

Text:
{text}
"""
    return call_openrouter(prompt)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    test_text = (
        "OpenRouter is an amazing AI service that makes building apps super easy. "
        "I love how fast it is!"
    )

    summary = summarize_text(test_text)
    sentiment = analyze_sentiment(test_text)

    print("=== Summary ===")
    print(summary)
    print("\n=== Sentiment ===")
    print(sentiment)

