import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🎨 AI Image Generator")

st.write(
    "Generate AI images using Hugging Face Inference API"
)

# ---------------------------------------------------
# LOAD SECRET TOKEN
# ---------------------------------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

# ---------------------------------------------------
# API URL
# ---------------------------------------------------
API_URL = (
    "https://api-inference.huggingface.co/models/"
    "runwayml/stable-diffusion-v1-5"
)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
prompt = st.text_input(
    "Enter your prompt",
    "A futuristic city at sunset"
)

# ---------------------------------------------------
# GENERATE BUTTON
# ---------------------------------------------------
if st.button("Generate Image"):

    with st.spinner("Generating image..."):

        payload = {
            "inputs": prompt
        }

        try:

            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=180
            )

            if response.status_code == 200:

                image = Image.open(
                    BytesIO(response.content)
                )

                st.image(
                    image,
                    caption=prompt,
                    use_container_width=True
                )

            elif response.status_code == 503:

                st.warning(
                    "Model is loading.\n"
                    "Wait 30 seconds and try again."
                )

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.write(response.text)

        except requests.exceptions.ConnectionError:

            st.error(
                "Connection failed."
            )

        except requests.exceptions.Timeout:

            st.error(
                "Request timeout."
            )

        except Exception as e:

            st.error(str(e))
