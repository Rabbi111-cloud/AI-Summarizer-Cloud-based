// Replace this with your Render backend URL:
const BASE_URL = "https://ai-summarizer-cloud-based-14.onrender.com";

function getText() {
    return document.getElementById("inputText").value.trim();
}

function showOutput(text) {
    document.getElementById("outputBox").textContent = text;
}

async function summarize() {
    const text = getText();
    if (!text) return showOutput("Please enter some text.");

    showOutput("Summarizing...");

    const res = await fetch(`${BASE_URL}/summarize`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text })
    });

    const data = await res.json();
    showOutput(JSON.stringify(data, null, 2));
}

async function sentiment() {
    const text = getText();
    if (!text) return showOutput("Please enter some text.");

    showOutput("Analyzing sentiment...");

    const res = await fetch(`${BASE_URL}/sentiment`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text })
    });

    const data = await res.json();
    showOutput(JSON.stringify(data, null, 2));
}

async function analyzeBoth() {
    const text = getText();
    if (!text) return showOutput("Please enter some text.");

    showOutput("Running full analysis...");

    const res = await fetch(`${BASE_URL}/analyze`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text })
    });

    const data = await res.json();
    showOutput(JSON.stringify(data, null, 2));
}
