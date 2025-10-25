import streamlit as st
import requests
import json
from PIL import Image
from io import BytesIO
import os

# --- 1. CONFIGURATION ---
# 🛑 CRITICAL: This URL MUST match your live Render endpoint URL.
API_URL = "https://grain-classifier-api.onrender.com/predict" 

GRAIN_INFO = {
    "Bajra": {"Protein (g)": "11.0", "Calories (per 100g)": "378", "Uses": "Flour for flatbreads (rotis), porridge, animal feed. Highly drought-resistant."},
    "Barley": {"Protein (g)": "12.5", "Calories (per 100g)": "352", "Uses": "Brewing (malt), soups, stews, animal fodder. Excellent source of fiber."},
    "Foxtail millet": {"Protein (g)": "12.3", "Calories (per 100g)": "331", "Uses": "Porridge, staple food in some areas, used in bird seed mixes."},
    "Jowar": {"Protein (g)": "10.6", "Calories (per 100g)": "349", "Uses": "Flatbreads (Jowar roti), syrup production, beer. Gluten-free alternative."},
    "Kodo millet": {"Protein (g)": "8.3", "Calories (per 100g)": "353", "Uses": "Substituted for rice, used in diabetes management due to low glycemic index."},
    "Little millet": {"Protein (g)": "7.7", "Calories (per 100g)": "341", "Uses": "Substitute for rice and semolina. Highly nutritious, quick-cooking."},
    "Maize": {"Protein (g)": "9.2", "Calories (per 100g)": "365", "Uses": "Cornmeal, oil, syrup, ethanol, animal feed. Staple crop globally."},
    "Proso": {"Protein (g)": "12.5", "Calories (per 100g)": "364", "Uses": "Fodder, birdseed, gluten-free flour. Known for fast growth."},
    "Ragi": {"Protein (g)": "7.3", "Calories (per 100g)": "385", "Uses": "Roti, porridge, beverages. Rich source of calcium and fiber."},
    "Rice": {"Protein (g)": "7.1", "Calories (per 100g)": "360", "Uses": "Primary global staple food, used in brewing (sake), various desserts."},
    "Wheat": {"Protein (g)": "13.0", "Calories (per 100g)": "340", "Uses": "Flour for bread, pasta, cakes, and other baked goods. Most widely grown crop."},
}

# --- 2. INFERENCE CALL ---
def call_api_predict(uploaded_file):
    """Sends the image to the FastAPI server for prediction."""
    
    # Reset file pointer and convert to bytes
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    try:
        response = requests.post(API_URL, files=files)
        response.raise_for_status() # Raise exception for 4XX or 5XX errors
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: The server could not process the request. Status Code: {e.response.status_code}")
        st.warning("Ensure the Render API is live and the model file is accessible.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("API Connection Error. Ensure the FastAPI server is running and the URL is correct.")
        return None


# --- 3. STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="AI Grain Classifier", layout="wide")

st.title("🌾 AI Grain Identifier")

st.markdown("""
### 📸 Photo Instructions (Crucial for 95% Accuracy!)
1. **Place only ONE grain** on a **clean, white paper.**
2. **Center the grain** in the middle of the frame.
3. Ensure **bright, even lighting**.
""")

uploaded_file = st.file_uploader(
    "Choose a photo from your gallery or take a new one", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image using PIL for display
    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    # --- PREDICT ---
    with st.spinner("Analyzing grain features..."):
        result = call_api_predict(uploaded_file)

    if result:
        # Prediction result structure is guaranteed by the API
        predicted_class = result['prediction']
        confidence = result['confidence']
        top_results = result['top_results']
        
        # Use the confidence score to determine color
        if confidence > 0.90:
            result_type = "success"
        elif confidence > 0.75:
            result_type = "warning"
        else:
            result_type = "error"

        # --- DISPLAY RESULTS ---
        with col2:
            st.header("🔍 Classification Result")
            st.markdown(f"**Predicted Grain:** <span style='font-size: 28px; color: {'green' if result_type == 'success' else 'orange'};'>{predicted_class}</span>", unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence * 100:.2f}%")

            if predicted_class in GRAIN_INFO:
                info = GRAIN_INFO[predicted_class]
                st.subheader("🌾 Nutritional Summary")
                
                metric_cols = st.columns(2)
                metric_cols[0].metric("Protein", f"{info['Protein (g)']} g")
                metric_cols[1].metric("Calories", f"{info['Calories (per 100g)']}")
                
                st.subheader("💡 Common Uses")
                st.markdown(info['Uses'])
            else:
                st.warning(f"Nutritional data for '{predicted_class}' is missing.")
            
            # Display Top 3 probabilities
            st.subheader("Top Guesses")
            for res in top_results:
                st.text(f"{res['class']}: {res['confidence'] * 100:.2f}%")