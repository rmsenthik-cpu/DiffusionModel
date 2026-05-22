import streamlit as st
from transformers import pipeline
from PIL import Image
import io

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Lightweight AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Lightweight AI Image Generator")

st.write(
    "Generate images using a lightweight CPU-friendly model."
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():

    generator = pipeline(
        "text-to-image",
        model="hf-internal-testing/tiny-stable-diffusion-pipe"
    )

    return generator

# ---------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------
with st.spinner("Loading lightweight model..."):

    pipe = load_model()

st.success("Model Loaded Successfully!")

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
prompt = st.text_input(
    "Enter your prompt",
    "A beautiful mountain landscape"
)

# ---------------------------------------------------
# GENERATE BUTTON
# ---------------------------------------------------
if st.button("Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
        st.stop()

    with st.spinner("Generating image..."):

        try:

            result = pipe(prompt)

            image = result["images"][0]

            # ---------------------------------------------------
            # DISPLAY IMAGE
            # ---------------------------------------------------
            st.image(
                image,
                caption=prompt,
                use_container_width=True
            )

            # ---------------------------------------------------
            # SAVE IMAGE
            # ---------------------------------------------------
            buffer = io.BytesIO()

            image.save(buffer, format="PNG")

            st.download_button(
                label="⬇ Download Image",
                data=buffer.getvalue(),
                file_name="generated_image.png",
                mime="image/png"
            )

        except Exception as e:

            st.error(f"Error: {str(e)}")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("ℹ About")

st.sidebar.info(
    """
    Lightweight AI Image Generator

    Features:
    - No GPU required
    - CPU friendly
    - Lightweight model
    - Streamlit compatible
    """
)
