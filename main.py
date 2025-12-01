import os
import requests

# Disable Gradio audio modules to avoid pyaudioop errors
os.environ["GRADIO_DISABLE_AUDIO"] = "True"

import gradio as gr

# -----------------------------
# Environment Variables
# -----------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY is not set. API calls will fail.")

MODEL = "openai/gpt-4o-mini"

# -----------------------------
# Summarization
# -----------------------------
def summarize_text(text):
    if not text:
        return "Error: No text provided."

    prompt = f"""
Summarize the following text into a short, clear paragraph:

{text}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=30)
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# Sentiment Analysis
# -----------------------------
def analyze_sentiment(text):
    if not text:
        return "Error: No text provided."

    prompt = f"""
Analyze the sentiment of the following text.

- Sentiment: Positive, Neutral, Negative
- Confidence score (0–1)
- One-sentence explanation

Text:
{text}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=30)
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# Gradio Interface
# -----------------------------
def gradio_interface(text):
    summary = summarize_text(text)
    sentiment = analyze_sentiment(text)
    return summary, sentiment

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=6, placeholder="Enter text here..."),
    outputs=[gr.Textbox(label="Summary"), gr.Textbox(label="Sentiment")],
    title="Cloud AI Summarizer & Sentiment Analyzer",
    description="Enter text to get summarized insights and sentiment analysis."
)

# -----------------------------
# Run on Render
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
