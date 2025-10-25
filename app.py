import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import os 

# --- 1. CONFIGURATION and API SETUP ---
# For local testing, keep this as localhost:8000
# FOR FINAL DEPLOYMENT, CHANGE THIS TO YOUR CLOUD SERVER URL (e.g., https://my-grain-api.com/predict)
API_URL = "http://localhost:8000/predict" 

# Nutritional and Use Information Lookup Table (CLEANED keys)
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


# --- 2. INFERENCE FUNCTION (API CALL) ---
def predict_via_api(image_bytes):
    """Sends image bytes to the FastAPI backend and retrieves prediction."""
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    
    try:
        # Check for placeholder API URL
        if 'localhost' not in API_URL and '127.0.0.1' not in API_URL:
            st.warning("Ensure your API server is running and the public URL is correct.")
            
        response = requests.post(API_URL, files=files)
        response.raise_for_status()  # Raise exception for 4xx or 5xx status codes
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"API Connection Error. Ensure the FastAPI server is running in a separate terminal.")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# --- 3. STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="AI Grain Classifier", layout="wide")

st.title("🌾 AI Grain Identifier")

st.markdown("""
### 📸 Photo Instructions (Crucial for 95% Accuracy!)
1. *Place only ONE grain* on a *clean, white paper.*
2. *Center the grain* in the middle of the frame (this replicates the model's training 'zoom').
3. Ensure *bright, even lighting* (e.g., natural daylight).
""")

uploaded_file = st.file_uploader(
    "Choose a photo from your gallery or take a new one", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Load image locally for display
        img = Image.open(uploaded_file)
        
        # Convert image to bytes for API transmission
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)

        # --- PREDICT ---
        with st.spinner("Sending image to server and analyzing features..."):
            api_result = predict_via_api(image_bytes)

        # --- DISPLAY RESULTS ---
        if api_result and 'prediction' in api_result:
            clean_class_name = api_result['prediction']
            confidence = api_result['confidence']
            
            if confidence > 0.90:
                status_emoji = "⭐"
            elif confidence > 0.75:
                status_emoji = "⚠"
            else:
                status_emoji = "❌"

            with col2:
                st.header("🔍 Classification Result")
                st.markdown(f"""
                    *Predicted Grain:* <span style='font-size: 28px; color: {'green' if confidence > 0.75 else 'red'};'>
                    {clean_class_name} {status_emoji}
                    </span>
                    """, unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence * 100:.2f}%")

                if clean_class_name in GRAIN_INFO:
                    info = GRAIN_INFO[clean_class_name]
                    st.subheader("🌾 Nutritional Summary")
                    
                    metric_cols = st.columns(2)
                    metric_cols[0].metric("Protein", f"{info['Protein (g)']} g")
                    metric_cols[1].metric("Calories", f"{info['Calories (per 100g)']}")
                    
                    st.subheader("💡 Common Uses")
                    st.markdown(info['Uses'])
                
                # Display Top Guesses
                st.subheader("Top Guesses")
                for item in api_result.get('top_results', []):
                    st.text(f"  - {item['class']}: {item['confidence'] * 100:.2f}%")

        
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")