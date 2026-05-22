import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Stable Diffusion Image Generator")
st.write("Generate AI images using Stable Diffusion")

# -------------------------------
# MODEL LOADING
# -------------------------------
@st.cache_resource
def load_model():

    model_id = "dreamlike-art/dreamlike-diffusion-1.0"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    return pipe

pipe = load_model()

# -------------------------------
# USER INPUT
# -------------------------------
prompt = st.text_input(
    "Enter your prompt",
    "AI loves nature"
)

generate = st.button("Generate Image")

# -------------------------------
# IMAGE GENERATION
# -------------------------------
if generate:

    with st.spinner("Generating image..."):

        image = pipe(prompt).images[0]

        st.image(image, caption=prompt, use_container_width=True)

        image.save("generated_image.png")

        with open("generated_image.png", "rb") as file:
            st.download_button(
                label="Download Image",
                data=file,
                file_name="generated_image.png",
                mime="image/png"
            )
