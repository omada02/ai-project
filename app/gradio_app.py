import gradio as gr
from src.predict import predict

def classify(image):
    image.save("temp.png")
    label = predict("temp.png")
    return f"Predicted class: {label}"

iface = gr.Interface(fn=classify, inputs="image", outputs="text")
iface.launch()
