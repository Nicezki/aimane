import gradio as gr
from app import aimane as Ai



# App will have 5 tab
# Status, Prepare, Train, Predict, Settings

ai = Ai.AiMane()

def get_status():
    status = ai.get_status()
    return status

def get_version():
    version = ai.get_version()
    return version


app = gr.Interface(fn=get_status, inputs=None, outputs="text", title="AiMane", description="AiMane is a tool for training and deploying machine learning models.")


app.launch()
