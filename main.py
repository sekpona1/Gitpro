

import numpy as np
import sys, os
from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
import io
import cv2
import pytesseract
import re
from pydantic import BaseModel


def read_img(img):
    pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
    text = pytesseract.image_to_string(img)
    return (text)


app = FastAPI()


class ImageType(BaseModel):
    url: str


@app.post("/ predict /")

def prediction(file: bytes = File(...)):
    image_stream = io.BytesIO(file)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    label = read_img(frame)
    return label






import numpy as np
from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request

import PIL
from PIL import Image, ImageOps
import numpy





import io
import cv2
import pytesseract
from pydantic import BaseModel
def read_img(img):
     pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
     text = pytesseract.image_to_string(img)
     return(text)


app = FastAPI()
class ImageType(BaseModel):
    url: str



@app.post("/predict/")
def prediction(file: bytes = File(...)):
    image_stream = io.BytesIO(file)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    label = read_img(frame)
    return label








"""import shutil
import aiofiles
from fastapi import FastAPI, UploadFile, File

from fastapi.responses import FileResponse
from PIL import Image
from io import BytesIO
import pytesseract as tess

#import yolo

tess.pytesseract.tesseract_cmd = r'C:\Users\sitsope sekpona\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'



def tesser(image):
    #img = Image.open(BytesIO(image))
    text = tess.image_to_string(image)
    return text


from fastapi import FastAPI, Request, File, UploadFile

from pydantic import BaseModel
import numpy as np
import cv2


app=FastAPI()

@app.get("/")
def home(request: Request):
    return {"hello": "world"}




@app.post("/extract_text")
async def extract_text(request: Request):
    label = ""
    if request.method == "POST":
        form = await request.form()
        # file = form["upload_file"].file
        contents = await form["upload_file"].read()
        image_stream = BytesIO(contents)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        label = tesser(frame)

    return {"label": label}
"""

"""
@app.post("/audio/")
async def voice(file: UploadFile= File(...)):
    out_file_path = f"{file.filename}"
    async with aiofiles.open(out_file_path, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write
    a=recog(f"{file.filename}")
    #v=aud.fonction(a)
    audio2=speech(a)
    return FileResponse(audio2)"""



"""@app.post('/yolo')
async def yolom(image: UploadFile = File(...)):
    temp_file = _save_file_to_disk(image)
    text =yolo.predict(temp_file)
    return text
"""
"""import aud
@app.post('/voice')
async def audio1(file: UploadFile= File(...)):
    out_file_path=f"{file.filename}"
    async with aiofiles.open(out_file_path, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write
    a=recog(f"{file.filename}")
    v=aud.fonction(a)
    #audio2=speech(v)
    return v #FileResponse(audio2)"""