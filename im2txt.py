import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
from io import BytesIO

def main():
    st.set_page_config(layout="wide") # Set page to wide layout
    st.title("Image To Text")
    image_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg', 'tiff'])

    if image_file is not None:
    # Error handling for file reading
        try:
            image = Image.open(image_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return 
        # Create two columns
        col1, col2 = st.columns([1, 1]) # Adjust the ratio between the columns

        # Display the image in the first column
        col1.image(image, use_column_width=True)   
        # Display the extract text button in the second column
        if col2.button('Extract Text'):
            # Error handling for OCR processing
            try:
                reader = easyocr.Reader(['en'])
                result = reader.readtext(image_cv, detail=1)
            except Exception as e:
                st.error(f"Error extracting text: {e}")
                return

            # Convert the result to a formatted string
            formatted_text = ''
            for item in result:
                formatted_text += item[1] + '\n' # Add a newline character after each line of text 
            # Display the extracted text in a code block for easy copying
            col2.code(formatted_text, language="plain text")

if __name__ == "__main__":
    main()
