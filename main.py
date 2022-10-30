from typing import Union

from fastapi import FastAPI, File, UploadFile

from io import BytesIO

import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
import numpy as np
import pandas as pd

app = FastAPI()

model = None

def load_model():
    model = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/food_V1/1')
    return model

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()

    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / image.max()
    output = model(image)
    predicted_index = output.numpy().argmax()
    labelmap_url = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_food_V1_labelmap.csv"
    classes = list(pd.read_csv(labelmap_url)["name"])

    return {"class": classes[predicted_index]}

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)

    return prediction

