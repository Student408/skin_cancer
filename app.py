import pip

def install_requirements():
    pip.main(['install', '-r', 'requirements.txt'])

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

# Custom CSS for a minimal, iOS-inspired interface with dark mode support
st.markdown("""
<style>
    .reportview-container {
        background: var(--background-color);
        color: var(--text-color);
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        color: var(--text-color);
    }
    h1 {
        font-size: 1.8rem;
    }
    h2 {
        font-size: 1.4rem;
    }
    h3 {
        font-size: 1.2rem;
    }
    .stButton>button {
        color: white;
        background-color: #007AFF;
        border-radius: 20px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 1px solid var(--secondary-background-color);
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .stFileUploader>div>div>button {
        border-radius: 10px;
    }
    .css-1v0mbdj.ebxwdo61 {
        border-radius: 10px;
    }
    .stAlert {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-radius: 10px;
        border: none;
    }
    @media (max-width: 640px) {
        .main .block-container {
            padding-left: 0.2rem;
            padding-right: 0.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load models (assuming they are in the same directory as the script)
@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model("./model/keras_Model_2.16.1.h5")
    model2 = tf.keras.models.load_model("./model/mobilenetv2_model_v6.h5")
    model3 = tf.keras.models.load_model("./model/mobilenetv2_model_v6.h5")
    return model1, model2, model3

# Define class names and weights
class_names = ['Basal Cell Carcinoma', 'Melanoma', 'Nevus', 'Benign Keratosis', 'No Cancer']
weights_model1 = np.array([0.6, 0.6, 0.6, 0.6, 0.34])
weights_model2 = np.array([0.2, 0.2, 0.2, 0.2, 0.33])
weights_model3 = np.array([0.2, 0.2, 0.2, 0.2, 0.33])

# Preprocessing functions
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

# Combined prediction function
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

# Grad-CAM functions
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
    model.layers[-1].activation = None  # Remove last layer's softmax
    return make_gradcam_heatmap(img_array, model, last_conv_layer_name)

def apply_gradcam(img, heatmap, alpha=0.4):
    img = img.copy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# Main Streamlit app
def main():
    st.title("Skin Cancer Prediction")

    # Load models
    with st.spinner("Loading models... Please wait."):
        model1, model2, model3 = load_models()
    st.success("Models loaded successfully!")

    st.write("Upload an image or provide a URL for skin cancer prediction.")

    col1, col2 = st.columns(2)

    with col1:
        url = st.text_input("Image URL:", placeholder="https://example.com/image.jpg")
    
    with col2:
        uploaded_file = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])

    if url or uploaded_file:
        try:
            if url:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(uploaded_file).convert("RGB")
        
            # Resize the image to a maximum width of 400 pixels while maintaining aspect ratio
            max_width = 700
            ratio = max_width / float(img.size[0])
            height = int((float(img.size[1]) * float(ratio)))
            img = img.resize((max_width, height), Image.Resampling.LANCZOS)
        
            st.image(img, caption="Uploaded Image", use_column_width=True)
        
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image... Please wait."):
                    predicted_class, confidence, all_probabilities = combined_predict_skin_cancer(img, model1, model2, model3)
            
                st.subheader("Prediction Results")
                st.markdown(f"**Predicted condition:** {predicted_class}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
            
                # Create a DataFrame for the probabilities
                df = pd.DataFrame({
                    'Condition': class_names,
                    'Probability': all_probabilities
                })
                
                # Sort the DataFrame by probability in descending order
                df = df.sort_values('Probability', ascending=False)
                
                # Create a horizontal bar chart using Plotly
                colors = ['#FF3B30' if condition == predicted_class else '#007AFF' for condition in df['Condition']]
                colors = ['#34C759' if condition == 'No Cancer' else color for condition, color in zip(df['Condition'], colors)]

                fig = go.Figure()

                # Add bars
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

                # Update layout
                fig.update_layout(
                    title={
                        'text': 'Probability of Each Condition',
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': dict(size=16)
                    },
                    xaxis_title=None,  # Remove x-axis title
                    yaxis_title=None,  # Remove y-axis title
                    height=300,  # Reduce height for better mobile view
                    margin=dict(l=10, r=10, t=40, b=10),  # Tighten margins
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000', size=12),
                    showlegend=False,
                    bargap=0.2,  # Adjust gap between bars
                )

                # Update axes
                fig.update_xaxes(
                    range=[0, 100],  # Set range from 0 to 100%
                    visible=False,  # Hide x-axis
                )
                fig.update_yaxes(
                    color='#FFFFFF' if st.get_option("theme.base") == "dark" else '#000000',
                    tickfont=dict(size=12),
                    automargin=True,  # Automatically adjust margins to fit labels
                )

                # Display the chart
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': False,  # Hide the mode bar
                    'responsive': True
                })

                # Add a legend explaining the color coding
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

        except Exception as e:
            st.error(f"An error occurred: {str(e)}. Please try again with a different image.")

    with st.sidebar:
        st.header("About")
        st.info("""
        This app uses AI to analyze skin images and predict various conditions, including skin cancer.

        It can detect:
        - Basal Cell Carcinoma
        - Melanoma
        - Nevus
        - Benign Keratosis
        - No Cancer (healthy skin)

        **Disclaimer:** This app is for educational purposes only. Always consult a healthcare professional for proper diagnosis and treatment.
        """)

        st.header("How to Use")
        st.markdown("""
        1. Enter an image URL or upload an image file.
        2. Click "Analyze Image".
        3. View the prediction results and probability breakdown.
        4. For concerning results, a heatmap will highlight areas of interest.
        5. Always consult with a healthcare professional for proper diagnosis.
        """)

if __name__ == "__main__":
    main()