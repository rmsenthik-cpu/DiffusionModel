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
# LOAD HF TOKEN
# ---------------------------------------------------
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]

except Exception as e:
    st.error("HF_TOKEN missing in Streamlit Secrets")
    st.stop()

# ---------------------------------------------------
# WORKING MODEL API
# ---------------------------------------------------
API_URL = (
    "https://api-inference.huggingface.co/models/"
    "stabilityai/stable-diffusion-2"
)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ---------------------------------------------------
# PROMPT INPUT
# ---------------------------------------------------
prompt = st.text_input(
    "Enter Prompt",
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
                timeout=300
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
                    "Model is loading on Hugging Face.\n"
                    "Wait 30-60 seconds and try again."
                )

            # UNAUTHORIZED
            elif response.status_code == 401:

                st.error(
                    "Invalid Hugging Face Token."
                )

            # OTHER API ERRORS
            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.write(response.text)

        except requests.exceptions.Timeout:

            st.error(
                "Request timed out."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Connection failed.\n"
                "Cannot reach Hugging Face API."
            )

        except Exception as e:

            st.error(str(e))
