import importlib
import subprocess
import sys

def install_requirements():
    try:
        importlib.import_module('numpy')
        print("Requirements are already installed.")
    except ImportError:
        print("Installing requirements...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

install_requirements()

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import requests
from io import BytesIO
import cv2
import pandas as pd
import plotly.graph_objects as go

# Set page config at the very beginning
st.set_page_config(page_title="Skin Cancer Prediction", layout="wide", page_icon="🩺")

# Custom CSS (same as before)
st.markdown("""
<style>
    /* Your custom CSS here */
</style>
""", unsafe_allow_html=True)

# Load models (same as before)
@st.cache_resource
def load_models():
    # model1 = tf.keras.models.load_model("./model/keras_Model_2.16.1.h5")
    model1 = tf.keras.models.load_model("./model/keras_Model_tm4kv2_2.16.1.h5")
    model2 = tf.keras.models.load_model("./model/mobilenetv2_model_v6.h5")
    model3 = model2
    return model1, model2, model3

# Define class names and weights (same as before)
class_names = ['Basal Cell Carcinoma', 'Melanoma', 'Nevus', 'Benign Keratosis', 'No Cancer']
weights_model1 = np.array([0.6, 0.6, 0.6, 0.6, 0.5])
weights_model2 = np.array([0.2, 0.2, 0.2, 0.2, 0.25])
weights_model3 = np.array([0.2, 0.2, 0.2, 0.2, 0.25])

# Preprocessing functions (same as before)
def preprocess_image_model1(img):
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    img_array = (img_array.astype(np.float32) / 127.5) - 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_model2(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype('float32') / 255
    return img_array

preprocess_image_model3 = preprocess_image_model2

# Combined prediction function (same as before)
def combined_predict_skin_cancer(img, model1, model2, model3):
    processed_image1 = preprocess_image_model1(img)
    processed_image2 = preprocess_image_model2(img)
    processed_image3 = preprocess_image_model3(img)

    predictions1 = model1.predict(processed_image1)[0]
    predictions2 = model2.predict(processed_image2)[0]
    predictions3 = model3.predict(processed_image3)[0]

    weighted_predictions1 = predictions1 * weights_model1
    weighted_predictions2 = predictions2 * weights_model2
    weighted_predictions3 = predictions3 * weights_model3

    combined_predictions = weighted_predictions1 + weighted_predictions2 + weighted_predictions3
    combined_predictions = combined_predictions / np.sum(combined_predictions)

    predicted_class_index = np.argmax(combined_predictions)
    predicted_class = class_names[predicted_class_index]
    confidence = combined_predictions[predicted_class_index] * 100
    probabilities_percentage = combined_predictions * 100

    return predicted_class, confidence, probabilities_percentage

# Grad-CAM functions (same as before)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def generate_heatmap(img_array, model, last_conv_layer_name):
    img_array = preprocess_image_model3(Image.fromarray(img_array))
    model.layers[-1].activation = None
    return make_gradcam_heatmap(img_array, model, last_conv_layer_name)

def apply_gradcam(img, heatmap, alpha=0.4):
    img = img.copy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# Class information dictionary (same as before)
class_info = {
    'Basal Cell Carcinoma': {
        'description': 'Basal cell carcinoma is a type of skin cancer that begins in the basal cells.',
        'wiki_link': 'https://en.wikipedia.org/wiki/Basal-cell_carcinoma'
    },
    'Melanoma': {
        'description': 'Melanoma is the most serious type of skin cancer that develops in melanocytes.',
        'wiki_link': 'https://en.wikipedia.org/wiki/Melanoma'
    },
    'Nevus': {
        'description': 'A nevus (or mole) is a benign growth of melanocytes.',
        'wiki_link': 'https://en.wikipedia.org/wiki/Nevus'
    },
    'Benign Keratosis': {
        'description': 'Benign keratosis refers to non-cancerous skin growths, typically caused by sun exposure.',
        'wiki_link': 'https://en.wikipedia.org/wiki/Seborrheic_keratosis'
    },
    'No Cancer': {
        'description': 'No signs of cancer were detected in the analyzed image.',
        'wiki_link': 'https://en.wikipedia.org/wiki/Skin_cancer'
    }
}

# Main Streamlit app
def main():
    st.title("Skin Cancer Prediction - Multiple Patients")

    # Load models
    with st.spinner("Loading models... Please wait."):
        model1, model2, model3 = load_models()
    st.success("Models loaded successfully!")

    st.write("Enter image URLs with patient names or upload multiple images.")

    # Input for multiple URLs and patient names
    url_input = st.text_area("Enter image URLs and patient names (one per line, separated by comma):", 
                             placeholder="https://example.com/image1.jpg, John Doe\nhttps://example.com/image2.jpg, Jane Smith")

    # File uploader for multiple images
    uploaded_files = st.file_uploader("Upload images:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Process URLs
    url_data = []
    if url_input:
        for line in url_input.split('\n'):
            if ',' in line:
                url, name = line.split(',')
                url_data.append((url.strip(), name.strip()))

    # Process uploaded files
    file_data = []
    if uploaded_files:
        for file in uploaded_files:
            file_data.append((file, file.name))

    # Combine both sources of images
    all_images = url_data + file_data

    if all_images:
        results = []
        for img_source, patient_name in all_images:
            try:
                if isinstance(img_source, str):  # URL
                    response = requests.get(img_source)
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                else:  # File
                    img = Image.open(img_source).convert("RGB")

                predicted_class, confidence, _ = combined_predict_skin_cancer(img, model1, model2, model3)
                results.append({
                    "Patient Name": patient_name,
                    "Predicted Condition": predicted_class,
                    "Confidence": f"{confidence:.2f}%"
                })

            except Exception as e:
                st.error(f"Error processing image for {patient_name}: {str(e)}")

        # Display results in a table
        if results:
            st.subheader("Prediction Results")
            df = pd.DataFrame(results)
            st.table(df)

        # Option to show detailed analysis for a specific patient
        if len(results) > 0:
            selected_patient = st.selectbox("Select a patient for detailed analysis:", 
                                            [result["Patient Name"] for result in results])
            
            if st.button("Show Detailed Analysis"):
                # Find the selected patient's data and image
                selected_data = next((item for item in all_images if item[1] == selected_patient), None)
                if selected_data:
                    img_source, _ = selected_data
                    if isinstance(img_source, str):  # URL
                        response = requests.get(img_source)
                        img = Image.open(BytesIO(response.content)).convert("RGB")
                    else:  # File
                        img = Image.open(img_source).convert("RGB")

                    # Display the image and perform detailed analysis
                    st.image(img, caption=f"Image for {selected_patient}", use_column_width=True)
                    
                    predicted_class, confidence, all_probabilities = combined_predict_skin_cancer(img, model1, model2, model3)
                    
                    st.subheader("Detailed Analysis")
                    st.markdown(f"**Predicted condition:** {predicted_class}")
                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                    
                    if predicted_class in class_info:
                        st.markdown(f"**Information about {predicted_class}:** {class_info[predicted_class]['description']}")
                        st.markdown(f"[Learn more on Wikipedia]({class_info[predicted_class]['wiki_link']})")

                    # Create a DataFrame for the probabilities
                    df = pd.DataFrame({
                        'Condition': class_names,
                        'Probability': all_probabilities
                    })
                    
                    df = df.sort_values('Probability', ascending=False)
                    
                    # Create a horizontal bar chart using Plotly
                    colors = ['#FF3B30' if condition == predicted_class else '#007AFF' for condition in df['Condition']]
                    colors = ['#34C759' if condition == 'No Cancer' else color for condition, color in zip(df['Condition'], colors)]

                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        y=df['Condition'],
                        x=df['Probability'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{prob:.1f}%" for prob in df['Probability']],
                        textposition='inside',
                        insidetextanchor='start',
                        textfont=dict(color='white', size=12),
                        hoverinfo='text',
                        hovertext=[f"{cond}: {prob:.2f}%" for cond, prob in zip(df['Condition'], df['Probability'])]
                    ))

                    fig.update_layout(
                        title={
                            'text': 'Probability of Each Condition',
                            'y':0.95,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': dict(size=16)
                        },
                        xaxis_title=None,
                        yaxis_title=None,
                        height=300,
                        margin=dict(l=10, r=10, t=40, b=10),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000', size=12),
                        showlegend=False,
                        bargap=0.2,
                    )

                    fig.update_xaxes(
                        range=[0, 100],
                        visible=False,
                    )
                    fig.update_yaxes(
                        color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000',
                        tickfont=dict(size=12),
                        automargin=True,
                    )

                    st.plotly_chart(fig, use_container_width=True, config={
                        'displayModeBar': False,
                        'responsive': True,
                        'scrollZoom': False,
                        'doubleClick': False,
                        'showTips': False,
                        'dragmode': False,
                        'staticPlot': True
                    })

                    st.markdown("""
                    <div style="display: flex; justify-content: space-around; font-size: 0.8em;">
                        <span><span style='color: #FF3B30;'>■</span> Predicted</span>
                        <span><span style='color: #007AFF;'>■</span> Other</span>
                        <span><span style='color: #34C759;'>■</span> No Cancer</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                    if predicted_class != "No Cancer":
                        st.subheader("Areas of Interest")
                        st.write("This heatmap highlights the areas of the image that were most influential in the model's prediction.")
                        img_array = np.array(img)
                        heatmap = generate_heatmap(img_array, model3, "block_16_depthwise")
                        superimposed_img = apply_gradcam(img_array, heatmap)
                        st.image(superimposed_img, caption="Grad-CAM Heatmap", use_column_width=True)
                    else:
                        st.info("No areas of concern detected.")

                    st.warning("Remember: This analysis is not a substitute for professional medical advice. If you have any concerns about your skin health, please consult a dermatologist or healthcare provider.")

    else:
        st.info("Please enter image URLs with patient names or upload images to begin analysis.")

        with st.sidebar:
            st.header("About")
            st.info("""
            This app uses AI to analyze skin images and predict various conditions, including skin cancer.

        It can detect:
        - Basal Cell Carcinoma
        - Melanoma
        - Nevus
        - Benign Keratosis
        - No Cancer

        **Disclaimer:** This app is for educational purposes only. Always consult a healthcare professional for proper diagnosis and treatment.
        """)

            st.header("How to Use")
            st.markdown("""
        1. Enter image URLs with patient names (one per line, separated by comma) or upload multiple image files.
        2. View the summary table of predictions for all patients.
        3. Select a patient from the dropdown menu for detailed analysis.
        4. Click "Show Detailed Analysis" to view in-depth results for the selected patient.
        5. For concerning results, a heatmap will highlight areas of interest.
        6. Always consult with a healthcare professional for proper diagnosis.
        """)

            st.markdown("---")
            st.markdown("### Credits")
            st.markdown("[GitHub](https://github.com/Student408/skin_cancer) | [LinkedIn](https://www.linkedin.com/in/ranjanshettigar)")

if __name__ == "__main__":
    main()
