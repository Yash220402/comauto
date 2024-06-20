import cv2
import numpy as np
import pytesseract
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load pre-trained OCR models
tesseract_lang = 'eng'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load Tesseract-based OCR model
ocr_model = AutoModelForSequenceClassification.from_pretrained('tesseract-base')
ocr_tokenizer = AutoTokenizer.from_pretrained('tesseract-base')

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance text
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphology to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    return opening

def ocr_tesseract(image):
    # Run Tesseract OCR
    text = pytesseract.image_to_string(Image.fromarray(image), lang=tesseract_lang)

    return text

def ocr_ocr_model(image):
    # Preprocess image for OCR model
    image = Image.fromarray(image)
    image = image.convert('RGB')
    image = image.resize((640, 480))

    # Encode image
    inputs = ocr_tokenizer(image, return_tensors='pt')

    # Run OCR model
    outputs = ocr_model(**inputs)
    text = torch.argmax(outputs.logits, dim=1).item()

    return ocr_tokenizer.decode(text, skip_special_tokens=True)

def ocr_pipeline(image):
    # Preprocess image
    preprocessed_image = preprocess_image(image)

    # Run multiple OCR models
    tesseract_text = ocr_tesseract(preprocessed_image)
    ocr_model_text = ocr_ocr_model(preprocessed_image)

    # Combine OCR outputs
    combined_text = '\n'.join([tesseract_text, ocr_model_text])

    return combined_text

# Test the OCR pipeline
image = cv2.imread('low_quality_invoice_image.png')
print(ocr_pipeline(image))
