from PIL import Image, ImageDraw
import cv2
from pathlib import Path
import easyocr
import matplotlib.pyplot as plt
from pprint import pprint as pp
from tqdm import tqdm
import json


def create_bounding_box(bbox_data):
    xs = []
    ys = []
    for x, y in bbox_data:
        xs.append(x)
        ys.append(y)
    
    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))
    
    return [left, top, right, bottom]

image_path = "invoice4.jpg"
image = Image.open(image_path)
width, height = image.size
scale = 1
image = image.resize((int(width*scale), int(height*scale)))
image.save(str(image_path))

reader = easyocr.Reader(['en'])
ocr_result = reader.readtext(str(image_path))

# Create rectanlges
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 28))

left_image = Image.open(image_path).convert("RGB")
right_image = Image.new("RGB", left_image.size, (255, 255, 255))

left_draw = ImageDraw.Draw(left_image)
right_draw = ImageDraw.Draw(right_image)

for i, (bbox, word, confidence) in enumerate(ocr_result):
    box = create_bounding_box(bbox)

    left_draw.rectangle(box, outline="blue", width=2)
    left, top, right, bottom = box

    left_draw.text((right + 5, top), text=str(i + 1), fill="red") 
    right_draw.text((left, top), text=word, fill="black")
    
ax1.imshow(left_image)
ax2.imshow(right_image)
ax1.axis("off");
ax2.axis("off");


ocr_result = reader.readtext(str(image_path), batch_size=16)
ocr_page = []
for bbox, word, confidence in ocr_result:
    ocr_page.append({
        "word": word, "bounding_box": create_bounding_box(bbox)
    })

for i in range(len(ocr_page)):
    print(ocr_page[i])

plt.show()
