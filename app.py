import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨"
)

st.title("🎨 AI Image Generator")

# ---------------------------------------------------
# LOAD SECRET TOKEN
# ---------------------------------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]

# ---------------------------------------------------
# MODEL API
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

        try:

            payload = {
                "inputs": prompt
            }

            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=180
            )

            # SUCCESS
            if response.status_code == 200:

                image = Image.open(
                    BytesIO(response.content)
                )

                st.image(
                    image,
                    caption=prompt,
                    use_container_width=True
                )

            # MODEL LOADING
            elif response.status_code == 503:

                st.warning(
                    "Model is loading. "
                    "Wait 30 seconds and try again."
                )

            else:

                st.error(
                    f"Error {response.status_code}"
                )

                st.write(response.text)

        except requests.exceptions.ConnectionError:

            st.error(
                "Connection failed."
            )

        except requests.exceptions.Timeout:

            st.error(
                "Request timed out."
            )

        except Exception as e:

            st.error(str(e))
