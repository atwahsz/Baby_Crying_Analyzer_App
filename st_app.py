import streamlit as st
import librosa
import torch
from transformers import AutoConfig, AutoModelForAudioClassification, AutoFeatureExtractor
import soundfile as sf
from safetensors.torch import load_file as safe_torch_load
import os
import sounddevice as sd
import numpy as np


# Load the model configuration
config = AutoConfig.from_pretrained("jstoone/distil-ast-audioset-finetuned-cry")

# Load the model and feature extractor
model = AutoModelForAudioClassification.from_pretrained("jstoone/distil-ast-audioset-finetuned-cry", config=config)
feature_extractor = AutoFeatureExtractor.from_pretrained("jstoone/distil-ast-audioset-finetuned-cry")

# Function to load and preprocess the audio file
def preprocess_audio(file_path):
    audio, sampling_rate = librosa.load(file_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    return inputs

# Function to compute the model output
def compute_model_output(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs

# Main function to load the audio file and compute the model output
def main(file_path):
    inputs = preprocess_audio(file_path)
    outputs = compute_model_output(inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return predicted_label

# Dictionary for results
reasons = {
    0: "بطنه يعوره فيه مغص",
    1: "فيه غازات",
    2: "متضايق او طفشان غيروله",
    3: "جوعان",
    4: "تعبان يبي ينام"
}

# Dictionary for GIFs
gifs = {
    0: "GIF_SHOWING_BABY_CRYING.GIF",
    1: "GIF_SHOWING_BABY_CRYING.GIF",
    2: "GIF_SHOWING_BABY_CRYING.GIF",
    3: "GIF_SHOWING_HUNGREY_BABY.GIF",
    4: "GIF_SHOWING_SLEEPY_BABY.GIF"
}

# Streamlit app
st.image("GIF_SHOWING_CARE_MOM_BABY.GIF")
st.title('برنامج تحليل بكاء الطفل')
st.markdown("""
<style>
.big-font {
    font-size:50px !important;
    font-weight: bold;
    color: #FF69B4;
}

body {
    background-color: #F0F8FF;
}

.stRadio > label {
    color: #FF69B4;
}

.stButton > button {
    background-color: #FF69B4;
    color: white;
}
</style>
""", unsafe_allow_html=True)

option = st.radio('اختر طريقة تحميل الصوت', ('رفع ملف الصوت', 'تسجيل صوت جديد'))

if option == 'رفع ملف الصوت':
    uploaded_file = st.file_uploader("رفع ملف الصوت", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_file_path = "temp_file." + uploaded_file.name.split(".")[-1]
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # If the file is not WAV, convert it to WAV
        if not temp_file_path.endswith(".wav"):
            audio, sampling_rate = librosa.load(temp_file_path, sr=16000)
            sf.write("temp_file.wav", audio, sampling_rate)
            temp_file_path = "temp_file.wav"

        # Classify the baby's cry
        predicted_label = main(temp_file_path)
        st.markdown(f'<p class="big-font">السبب المحتمل لبكاء الطفل هو: {reasons[predicted_label]}</p>', unsafe_allow_html=True)
        st.image(gifs[predicted_label])

        # Remove the temporary file
        os.remove(temp_file_path)

else:
    duration = 5  # Fixed duration of 5 seconds
    fs = 16000  # Sample rate

    if st.button('بدء التسجيل'):
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        st.write("جاري التسجيل...")
        sd.wait()  # Wait until recording is finished
        st.write("انتهى التسجيل.")

        # Save the recorded audio to a temporary location
        temp_file_path = "temp_file.wav"
        sf.write(temp_file_path, myrecording, fs)

        # Classify the baby's cry
        predicted_label = main(temp_file_path)
        st.markdown(f'<p class="big-font">السبب المحتمل لبكاء الطفل هو: {reasons[predicted_label]}</p>', unsafe_allow_html=True)
        st.image(gifs[predicted_label])

        # Remove the temporary file
        os.remove(temp_file_path)
