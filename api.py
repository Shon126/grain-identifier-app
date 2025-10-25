import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import uvicorn

# --- CONFIGURATION (Must match training) ---
MODEL_PATH = "mobilenet_final_best.keras"
IMAGE_SIZE = (224, 224)
CROP_FACTOR = 0.5  # 50% center crop
CLASS_NAMES = ['Bajra', 'Barley', 'Foxtail millet ', 'Jowar', 'Kodo millet', 'Little millet', 'Maize', 'Proso ', 'Ragi', 'Rice', 'Wheat ']

app = FastAPI(title="Grain Classifier API")
model = None

# --- Model Loading (Runs only once at startup) ---
@app.on_event("startup")
async def load_model():
    """Load the Keras model directly (skipping TFLite)."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            raise RuntimeError("Model file not found.")
            
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        print("Keras Model loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR loading Keras model: {e}")
        raise RuntimeError("Could not initialize Keras model.")

# --- Preprocessing Function (Simplified and Robust) ---
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Applies the exact processing pipeline: Crop, Resize, Normalize [-1, 1]."""
    
    # 1. Load image and convert to RGB
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # 2. Convert to NumPy array (float32) for TensorFlow processing
    img_np = np.array(img, dtype=np.float32) 
    img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)

    # 3. Automated Center Crop (The Zoom operation)
    cropped_img = tf.image.central_crop(img_tensor, central_fraction=CROP_FACTOR)
    resized_img = tf.image.resize(cropped_img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    # 4. Normalization to [-1, 1] range (MobileNetV2 expectation)
    normalized_tensor = (resized_img / 127.5) - 1.0

    # 5. Add Batch Dimension [1, H, W, C]
    input_data = tf.expand_dims(normalized_tensor, 0).numpy()
    
    return input_data

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict_grain(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    image_bytes = await file.read()
    
    try:
        input_tensor = preprocess_image(image_bytes)
        
        # Run Keras prediction directly
        predictions = model.predict(input_tensor, verbose=0)[0]
        probabilities = tf.nn.softmax(predictions).numpy()
        
        # Get top 3 results
        top_k = 3
        top_k_indices = np.argsort(probabilities)[::-1][:top_k]
        
        results = []
        for i in top_k_indices:
            results.append({
                "class": CLASS_NAMES[i].strip(),
                "confidence": float(probabilities[i])
            })
            
        return JSONResponse(content={
            "prediction": results[0]['class'],
            "confidence": results[0]['confidence'],
            "top_results": results
        })
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        # Note: If the error is prediction == 'Wheat' at 16%, the true error is in preprocessing
        raise HTTPException(status_code=500, detail=f"Prediction Error. Check input image quality. Details: {e}")

# (The server run command remains outside the file: uvicorn api:app --reload)