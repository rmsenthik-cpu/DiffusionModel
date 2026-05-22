import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Local AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Local AI Image Generator")

st.write("Generate AI images locally using Stable Diffusion")

# ---------------------------------------------------
# MODEL CACHE
# ---------------------------------------------------
@st.cache_resource
def load_model():

    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )

    pipe = pipe.to("cpu")

    return pipe

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
with st.spinner("Loading Stable Diffusion model..."):

    pipe = load_model()

st.success("Model Loaded Successfully!")

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
prompt = st.text_area(
    "Enter your image prompt",
    value="A futuristic city with flying cars at sunset",
    height=120
)

# ---------------------------------------------------
# IMAGE SIZE
# ---------------------------------------------------
width = st.slider(
    "Image Width",
    min_value=256,
    max_value=768,
    value=512,
    step=64
)

height = st.slider(
    "Image Height",
    min_value=256,
    max_value=768,
    value=512,
    step=64
)

# ---------------------------------------------------
# INFERENCE STEPS
# ---------------------------------------------------
steps = st.slider(
    "Inference Steps",
    min_value=10,
    max_value=50,
    value=20
)

# ---------------------------------------------------
# GENERATE BUTTON
# ---------------------------------------------------
if st.button("Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
        st.stop()

    with st.spinner("Generating image... Please wait..."):

        try:

            image = pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=steps
            ).images[0]

            st.image(
                image,
                caption=prompt,
                use_container_width=True
            )

            # ---------------------------------------------------
            # SAVE IMAGE
            # ---------------------------------------------------
            output_path = "generated_image.png"

            image.save(output_path)

            # ---------------------------------------------------
            # DOWNLOAD BUTTON
            # ---------------------------------------------------
            with open(output_path, "rb") as file:

                st.download_button(
                    label="⬇ Download Image",
                    data=file,
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
    This app uses:

    - Streamlit
    - Diffusers
    - Stable Diffusion v1.5
    - Local CPU inference

    No external API used.
    """
)

st.sidebar.warning(
    """
    ⚠ Streamlit Community Cloud may crash
    due to RAM limitations.

    Recommended:
    - Google Colab
    - RunPod
    - Local GPU machine
    """
)
