import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import random
import io

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Lightweight AI Art Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Lightweight AI Art Generator")

st.write(
    "Generate artistic AI-style images without GPU or heavy models."
)

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
prompt = st.text_input(
    "Enter your prompt",
    "Nature"
)

# ---------------------------------------------------
# GENERATE FUNCTION
# ---------------------------------------------------
def generate_art(prompt):

    width = 512
    height = 512

    image = Image.new(
        "RGB",
        (width, height),
        (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    )

    draw = ImageDraw.Draw(image)

    # Draw random circles
    for _ in range(100):

        x1 = random.randint(0, width)
        y1 = random.randint(0, height)

        x2 = x1 + random.randint(10, 100)
        y2 = y1 + random.randint(10, 100)

        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

        draw.ellipse(
            [x1, y1, x2, y2],
            fill=color
        )

    # Add prompt text
    try:
        font = ImageFont.load_default()

        draw.text(
            (20, 20),
            f"Prompt: {prompt}",
            fill="white",
            font=font
        )

    except:
        pass

    return image

# ---------------------------------------------------
# GENERATE BUTTON
# ---------------------------------------------------
if st.button("Generate Art"):

    with st.spinner("Generating artwork..."):

        image = generate_art(prompt)

        st.image(
            image,
            caption=prompt,
            use_container_width=True
        )

        # Download option
        buffer = io.BytesIO()

        image.save(buffer, format="PNG")

        st.download_button(
            label="⬇ Download Image",
            data=buffer.getvalue(),
            file_name="generated_art.png",
            mime="image/png"
        )

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("ℹ About")

st.sidebar.info(
    """
    Features:
    - No GPU
    - No torch
    - No diffusion
    - Lightweight
    - Streamlit Cloud compatible
    """
)
