

# This demo is created using Gradio - a Python library used to showcase ML model
import gradio as gr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model_dir = "./model"
processor = TrOCRProcessor.from_pretrained(model_dir , use_fast=True)  # Added use_fast=True
model = VisionEncoderDecoderModel.from_pretrained(model_dir)

def process_image(image):
    # prepare image
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    # generate (no beam search)
    generated_ids = model.generate(pixel_values)
    
    # decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text

title = "OCR Demo"

description = "Demo of Conversion of Handwritten Text into Editable Text."

# Use your actual local image files
examples = [["image_0.png"], ["image_1.png"], ["image_2.png"]]

iface = gr.Interface(fn=process_image, 
                     inputs=gr.Image(type="pil"), 
                     outputs=gr.Textbox(),
                     title=title,
                     description=description,
                     examples=examples)

iface.launch(debug=True)



