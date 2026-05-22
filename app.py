import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 AI Image Generator")

# ---------------------------------------
# LOAD SECRET API KEY
# ---------------------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

# ---------------------------------------
# HUGGING FACE MODEL API
# ---------------------------------------
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ---------------------------------------
# USER INPUT
# ---------------------------------------
prompt = st.text_input(
    "Enter your prompt",
    "AI loves nature"
)

generate = st.button("Generate Image")

# ---------------------------------------
# GENERATE IMAGE
# ---------------------------------------
if generate:

    with st.spinner("Generating image..."):

        payload = {
            "inputs": prompt
        }

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload
        )

        if response.status_code == 200:

            image = Image.open(BytesIO(response.content))

            st.image(
                image,
                caption=prompt,
                use_container_width=True
            )

            image.save("generated_image.png")

            with open("generated_image.png", "rb") as file:

                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )

        else:
            st.error(f"Error: {response.status_code}")
            st.write(response.text)
