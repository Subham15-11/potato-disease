from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("potato.h5")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
def ping():  # Removed async keyword since it's not necessary
    return "Hello, It is running"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        return {"class": predicted_class, "confidence": (float(confidence)*100)}
    except Exception as e:
        # Added error handling for reading the file and making predictions
        return {"error": str(e)}


if __name__ == "__main__":
    # Removed unnecessary uvicorn import
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)