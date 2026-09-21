import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-title {
        font-size: 28px;
        font-weight: bold;
    }

    .confidence {
        font-size: 22px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">♻️ Waste Classification AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning based waste segregation using MobileNetV2'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📌 About the Project")

    st.write(
        "This application uses a fine-tuned MobileNetV2 "
        "deep learning model to classify waste images."
    )

    st.subheader("Waste Categories")

    st.write("📦 Cardboard")
    st.write("🫙 Glass")
    st.write("🔩 Metal")
    st.write("📄 Paper")
    st.write("🧴 Plastic")
    st.write("🗑️ Trash")

    st.divider()

    st.subheader("🏆 Model Performance")

    st.metric(
        "Validation Accuracy",
        "≈ 87%"
    )

    st.metric(
        "Macro F1-Score",
        "0.85"
    )

    st.caption(
        "Model: Fine-tuned MobileNetV2"
    )

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "models/mobilenetv2_finetuned.keras"
    )


model = load_model()

# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# --------------------------------------------------
# WASTE INFORMATION
# --------------------------------------------------

waste_info = {

    "cardboard":
        "Place clean cardboard in the paper/cardboard recycling bin.",

    "glass":
        "Place glass containers in the appropriate glass recycling bin.",

    "metal":
        "Place metal cans and containers in the metal recycling bin.",

    "paper":
        "Place clean and dry paper in the paper recycling bin.",

    "plastic":
        "Place recyclable plastic containers in the plastic recycling bin.",

    "trash":
        "This item may require general waste disposal if it cannot be recycled."
}

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("📤 Upload a Waste Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # -------------------------------
    # IMAGE
    # -------------------------------

    with col1:

        st.subheader("🖼️ Uploaded Image")

        st.image(
            image,
            width=450
        )

    # -------------------------------
    # PREDICTION
    # -------------------------------

    image_resized = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image_resized
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # Do NOT apply preprocess_input here.
    # The fine-tuned model already contains
    # MobileNetV2 preprocessing.

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        predictions
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        predictions[predicted_index] * 100
    )

    # -------------------------------
    # RESULT
    # -------------------------------

    with col2:

        st.subheader("🔍 Prediction Result")

        st.success(
            f"Predicted Category: "
            f"{predicted_class.upper()}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            float(predictions[predicted_index])
        )

        st.info(
            waste_info[predicted_class]
        )

    # --------------------------------------------------
    # TOP 3 PREDICTIONS
    # --------------------------------------------------

    st.divider()

    st.subheader("📊 Top 3 Predictions")

    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    top_classes = [
        class_names[i].capitalize()
        for i in top_indices
    ]

    top_scores = [
        predictions[i] * 100
        for i in top_indices
    ]

    # Display percentages
    for i in range(3):

        st.write(
            f"**{i + 1}. {top_classes[i]}** "
            f"— {top_scores[i]:.2f}%"
        )

    # Bar chart
    chart_data = {
        "Category": top_classes,
        "Confidence (%)": top_scores
    }

    st.bar_chart(
        chart_data,
        x="Category",
        y="Confidence (%)"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Waste Classification AI | "
    "Deep Learning + MobileNetV2 + Streamlit"
)
