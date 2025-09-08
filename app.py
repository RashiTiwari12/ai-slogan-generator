import streamlit as st
import boto3
from botocore.config import Config
import time
import json
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Config from Secrets (with fallback)
# -----------------------------
RATE_LIMIT = st.secrets.get("rate_limit", {}).get("max_requests", 6)
TIME_WINDOW = st.secrets.get("rate_limit", {}).get("time_window_seconds", 60)
BEDROCK_REGION = st.secrets.get("aws", {}).get("region", "us-east-1")

# -----------------------------
# Rate Limiting Setup
# -----------------------------
def check_rate_limit():
    if "requests" not in st.session_state:
        st.session_state["requests"] = []

    now = time.time()
    st.session_state["requests"] = [
        ts for ts in st.session_state["requests"] if now - ts < TIME_WINDOW
    ]

    if len(st.session_state["requests"]) >= RATE_LIMIT:
        return False

    st.session_state["requests"].append(now)
    return True

# -----------------------------
# Bedrock Client Setup
# -----------------------------
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=Config(retries={"max_attempts": 3}),
)

# -----------------------------
# Function to extract title/description from URL
# -----------------------------
def extract_content_from_url(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            title = soup.title.string if soup.title else ""
            meta_desc = soup.find("meta", attrs={"name":"description"})
            description = meta_desc["content"] if meta_desc else ""
            return f"{title}. {description}".strip()
        else:
            return ""
    except Exception as e:
        return ""

# -----------------------------
# Streamlit Styling
# -----------------------------
st.set_page_config(
    page_title="AI Tagline Generator",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 50%, #ffc7e0 100%);
        font-family: 'Segoe UI', sans-serif;
        color: #111;
    }
    .title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #222;
        text-shadow: 1px 1px 3px rgba(255,255,255,0.5);
    }
    .subtitle {
        font-size: 1.25rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #333;
    }
    div[data-baseweb="input"] > input {
        border-radius: 12px;
        padding: 12px;
        font-size: 1rem;
        border: 2px solid #555;
        background: rgba(255,255,255,0.8);
        color: #111;
    }
    div.stButton > button {
        display: block;
        margin: 0 auto;
        background: linear-gradient(90deg, #ff9966 0%, #ff5e62 100%);
        color: #fff;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 15px;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff5e62 0%, #ff9966 100%);
        transform: scale(1.05);
    }
    .result-box {
        background: rgba(50,50,50,0.7);
        border-left: 5px solid #ffcc33;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        font-size: 1.6rem;
        font-weight: bold;
        text-align: center;
        color: #fff;
        box-shadow: 3px 3px 15px rgba(0,0,0,0.4);
        opacity: 0;
        animation: fadeIn 1s forwards;
    }
    @keyframes fadeIn { to { opacity: 1; } }
    .stError {
        background-color: rgba(255,0,0,0.2);
        border-left: 5px solid #ff0000;
        padding: 10px;
        border-radius: 8px;
        color: #111;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# App Content
# -----------------------------
st.markdown('<div class="title">AI Tagline Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter a product/idea and optional theme, or provide a URL to generate a tagline!</div>',
    unsafe_allow_html=True,
)

# Inputs
product = st.text_input("Enter your product/idea:")
theme = st.text_input("Optional: Enter the theme or style for the tagline:")
url = st.text_input("Or enter the URL of a lab/webpage:")

if st.button("Generate Tagline"):
    if not product and not url:
        st.error("Please enter a product/idea or a URL!")
    elif not check_rate_limit():
        st.error(
            f"Rate limit exceeded. Max {RATE_LIMIT} requests per {TIME_WINDOW} sec."
        )
    else:
        try:
            if url:
                content = extract_content_from_url(url)
                if not content:
                    st.error("Could not fetch content from the URL!")
                    st.stop()
                full_prompt = f"Generate a catchy tagline for the following page content: '{content}'"
            else:
                full_prompt = f"Generate a catchy tagline for the product/idea: '{product}'"
                if theme:
                    full_prompt += f" with the theme/style: '{theme}'"

            response = bedrock_client.invoke_model(
                modelId="amazon.nova-lite-v1:0",
                contentType="application/json",
                body=json.dumps(
                    {
                        "messages": [{"role": "user", "content": [{"text": full_prompt}]}],
                        "inferenceConfig": {"maxTokens": 60, "temperature": 0.8},
                    }
                ),
            )

            result_str = response["body"].read().decode("utf-8")
            result_json = json.loads(result_str)

            tagline = (
                (
                    result_json.get("output", {})
                    .get("message", {})
                    .get("content", [{}])[0]
                    .get("text", "No tagline generated")
                )
                .strip('"')
                .strip()
            )

            st.markdown(f'<div class="result-box">{tagline}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating tagline: {e}")
