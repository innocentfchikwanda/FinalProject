from keras.models import load_model
import streamlit as st
import pickle
import torch
import pandas as pd
import numpy as np
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
from keras.applications.xception import Xception
from keras.preprocessing.sequence import pad_sequences


def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Can't open image! Ensure that image path and extension is correct")

    image = image.resize((299,299))
    image = np.array(image)

    # for 4 channels images, we need to convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)

    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)

        if word is None:
            break

        in_text += ' ' + word

        if word == 'end':
            break

    return in_text




class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400,400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

def generate_image_and_save(prompt, model, output_path):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)

    # Save the image as a JPG file
    image.save(output_path, "JPEG")


with open('image_gen_model.pkl', 'rb') as imagefile:
  texttoimagemodel=pickle.load(imagefile)

with open('tokenizer.p', 'rb') as tokenfile:
  tokenizer=pickle.load(tokenfile)

import tensorflow as tf

# Replace 'model.h5' with the path to your HDF5 model file
imagetotextmodel = tf.keras.models.load_model("model_mtbest_221123.hdf5")



# Create a Streamlit app
st.title("Text - Image -Text")


feature1 = st.text_input("Enter text")


feature1 = str(feature1) 










if st.button("Generate"):
  output_path = r"ENter your output path and the name you would like the image to be saved as"
  generate_image_and_save(feature1, texttoimagemodel, output_path)


  img_path = r"Enter the same path as above"
  max_length = 29
  model = load_model(imagetotextmodel)
  xception_model = Xception(include_top=False, pooling="avg")
  photo = extract_features(img_path, xception_model)
  img = Image.open(img_path)
  description = generate_desc(model, tokenizer, photo, max_length)
  print("nn")
  st.write(description)
  st.write(plt.imshow(img))
