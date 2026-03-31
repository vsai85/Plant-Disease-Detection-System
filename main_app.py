# Library imports
import io
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf
from pathlib import Path

# Page config and styling
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

# Cache model to avoid reloading on every interaction
@st.cache_resource
def load_model_cached(path='plant_disease_model.h5'):
    return load_model(path)

# Loading the Model
model = load_model_cached('plant_disease_model.h5')
                    
# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Helper to make labels nicer
def pretty_label(s):
    try:
        name, disease = s.split('-', 1)
        disease = disease.replace('_', ' ')
        return f"{name} — {disease}"
    except Exception:
        return s.replace('_', ' ')

# Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Sidebar: app info and sample images
with st.sidebar:
    st.header("About")
    st.write("Upload a leaf photo and the model will predict the disease class.")
    st.write("Model loaded from: `plant_disease_model.h5`")
    show_summary = st.checkbox("Show model summary", False)
    show_raw = st.checkbox("Show raw predictions", False)

    # List sample images if available
    sample_images = []
    sample_dir = Path("Test Image")
    if sample_dir.exists():
        for p in sample_dir.iterdir():
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                sample_images.append(p)
    if sample_images:
        sample_choice = st.selectbox("Try a sample image", ["None"] + [p.name for p in sample_images])
        if sample_choice != "None":
            sample_path = sample_dir / sample_choice
        else:
            sample_path = None
    else:
        sample_path = None

# Optionally show model summary
if show_summary:
    with st.expander("Model summary"):
        buf = io.StringIO()
        model.summary(print_fn=lambda s: buf.write(s + "\n"))
        st.text(buf.getvalue())

# Page header and instructions
st.markdown("<h2 style='color:#2E8B57;'>Plant Disease Detection</h2>", unsafe_allow_html=True)
st.markdown("Upload a clear leaf photo (jpg/png). Try to center the leaf and avoid cluttered backgrounds.")

left, right = st.columns([1,1])

with left:
    st.subheader("Input")
    # If a sample is selected, use it; otherwise show uploader
    if sample_path:
        opencv_image = cv2.imread(str(sample_path))
        st.image(opencv_image, channels="BGR", caption=f"Sample: {sample_path.name}")
        st.write("Original image shape:", opencv_image.shape)
        uploaded = True
    else:
        plant_image = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
        uploaded = plant_image is not None
        if uploaded:
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(opencv_image, channels="BGR", caption="Uploaded image")
            st.write("Original image shape:", opencv_image.shape)

with right:
    st.subheader("Prediction")
    if not uploaded:
        st.info("Upload an image or pick a sample to enable prediction.")
    else:
        if st.button("Predict Disease", use_container_width=True):
            with st.spinner("Analyzing image..."):
                # Preprocess and predict
                img_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (256,256))
                input_img = img_resized.astype('float32') / 255.0
                input_batch = np.expand_dims(input_img, axis=0)

                try:
                    # Show model input shape when available
                    try:
                        st.write("Model input shape (expected):", model.input_shape)
                    except Exception:
                        pass

                    Y_pred = model.predict(input_batch)
                    if Y_pred.ndim == 2:
                        probs = tf.nn.softmax(Y_pred[0]).numpy()
                    else:
                        probs = tf.nn.softmax(Y_pred).numpy().ravel()

                    rows = [{"Label": pretty_label(CLASS_NAMES[i]), "Probability": float(probs[i])} for i in range(len(probs))]
                    rows = sorted(rows, key=lambda x: x["Probability"], reverse=True)
                    top = rows[0]
                    st.success(f"Top prediction: **{top['Label']}** — {top['Probability']:.4f}")

                    # Show a clean table
                    st.table([{"Label": r["Label"], "Probability": f"{r['Probability']:.4f}"} for r in rows])

                    # Visual progress bars
                    for r in rows:
                        c1, c2 = st.columns([2,5])
                        c1.write(r["Label"])
                        c2.progress(int(r["Probability"] * 100))

                    if show_raw:
                        st.write("Raw model predictions:", Y_pred)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.caption("Tip: If results look wrong, try several images or use the sample images for comparison.")