import streamlit as st
import boto3
from botocore.config import Config
import time
import json
import requests
from bs4 import BeautifulSoup
import re

# -----------------------------
# Config from Secrets
# -----------------------------
RATE_LIMIT = st.secrets.get("rate_limit", {}).get("max_requests", 6)
TIME_WINDOW = st.secrets.get("rate_limit", {}).get("time_window_seconds", 60)
BEDROCK_REGION = st.secrets.get("aws", {}).get("region", "us-east-1")

# -----------------------------
# Rate Limiting
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
# Bedrock Client
# -----------------------------
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=Config(retries={"max_attempts": 3}),
)

# -----------------------------
# Extract page content
# -----------------------------
def extract_content_from_url(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            # Remove site name/brand if present in title
            title = soup.title.string if soup.title else ""
            title = re.sub(r"[-|–|—].*$", "", title).strip()  # remove dash and trailing site name
            meta_desc = soup.find("meta", attrs={"name":"description"})
            description = meta_desc["content"] if meta_desc else ""
            return f"{title}. {description}".strip()
        else:
            return ""
    except Exception:
        return ""

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="AI Title Ideas", layout="centered")

st.title("AI Title Idea Generator")
st.write("Enter a URL of a lab or webpage. The AI will generate **creative title ideas** based on the content, ignoring the site name.")

url = st.text_input("Enter the URL of your lab/webpage:")

if st.button("Generate Title Ideas"):
    if not url:
        st.error("Please enter a URL!")
    elif not check_rate_limit():
        st.error(f"Rate limit exceeded. Max {RATE_LIMIT} requests per {TIME_WINDOW} sec.")
    else:
        try:
            content = extract_content_from_url(url)
            if not content:
                st.error("Could not fetch content from the URL!")
                st.stop()

            # Prompt: generate multiple title ideas, ignore site name
            prompt = (
                f"Generate 3-5 creative and catchy **title ideas** based on the following content. "
                f"Focus on keywords from the page and ignore the website name or branding:\n\n'{content}'"
            )

            response = bedrock_client.invoke_model(
                modelId="amazon.nova-lite-v1:0",
                contentType="application/json",
                body=json.dumps(
                    {
                        "messages": [{"role": "user", "content": [{"text": prompt}]}],
                        "inferenceConfig": {"maxTokens": 80, "temperature": 0.8},
                    }
                ),
            )

            result_str = response["body"].read().decode("utf-8")
            result_json = json.loads(result_str)

            titles = (
                result_json.get("output", {})
                .get("message", {})
                .get("content", [{}])[0]
                .get("text", "No title ideas generated")
            ).strip()

            st.subheader("Title Ideas:")
            st.write(titles)

        except Exception as e:
            st.error(f"Error generating title ideas: {e}")
