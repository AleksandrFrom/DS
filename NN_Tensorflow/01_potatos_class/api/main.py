from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

MODEL = tf.keras.models.load_model("models/model_potatos_val_25.keras")

CLASS_NAMES = ["Early Blinght", "Late Blight", "Healthy"]

@app.get('/ping')
async def ping():
    return 'Hello. It is work :)'

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
    
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)):
    
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    #requests.post()
    
    predictions = MODEL.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class' : predicted_class,
        'confidence': float(confidence)
    }
    
    


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)