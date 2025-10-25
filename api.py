import os
import numpy as np
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # 🛑 CRITICAL FIX: DISABLE GPU
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
import uvicorn
import cv2 
from PIL import Image

# --- CONFIGURATION (Must match training) ---
MODEL_PATH = "mobilenet_final_best.keras" # The large Keras file uploaded via LFS
IMAGE_SIZE = (224, 224)
CROP_FACTOR = 0.5  # 50% center crop for 2x zoom
CLASS_NAMES = ['Bajra', 'Barley', 'Foxtail millet ', 'Jowar', 'Kodo millet', 'Little millet', 'Maize', 'Proso ', 'Ragi', 'Rice', 'Wheat ']

app = FastAPI(title="Grain Classifier API")
model = None

# --- Model Loading (Runs only once at startup) ---
@app.on_event("startup")
async def load_model():
    """Load the Keras model directly on the CPU."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model file not found. Check Git LFS status.")
            
        # Loads on CPU due to CUDA_VISIBLE_DEVICES='-1'
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        print("Keras Model loaded successfully on CPU.")
    except Exception as e:
        print(f"FATAL ERROR loading Keras model: {e}")
        raise RuntimeError(f"Could not initialize Keras model. Error: {e}")


# --- CRITICAL PREPROCESSING FUNCTION ---
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Uses OpenCV to robustly decode image bytes, then applies Crop, Resize, Normalize [-1, 1]."""
    
    # 1. Decode image using NumPy and OpenCV (most reliable web decoding)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("Could not decode image bytes using OpenCV.")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Convert to TensorFlow float32 tensor
    img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)

    # 3. Automated Center Crop (The Zoom operation)
    cropped_img = tf.image.central_crop(img_tensor, central_fraction=CROP_FACTOR)
    resized_img = tf.image.resize(cropped_img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    # 4. Normalization to [-1, 1] range (Model expectation)
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
        # Step 1: Preprocess the input image using the robust OpenCV pipeline
        input_tensor = preprocess_image(image_bytes)
        
        # Step 2: Run prediction
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
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid image format or decoding error: {ve}")
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error. Details: {e}")