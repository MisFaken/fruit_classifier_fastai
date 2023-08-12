import gradio as gr
from fastai.vision.all import *
import skimage
import os 


learn_inf = load_learner('export.pkl')

title = "Fruit Classifier"
description = "A fruit classifier trained on a subset of the Fruit-262 dataset with fastai."
article="<p style='text-align: center'><a href='https://github.com/MisFaken/Clothing-segmentation-and-color-extraction-with-Self-Correction-Human-Parsing' target='_blank'> Github </a></p>"
examples = ['examples/' + f for f in os.listdir('examples') if f.endswith('.jpg')]

labels = learn_inf.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn_inf.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=3), title=title, description=description, article=article, examples=examples ,interpretation='default').launch(share=True)