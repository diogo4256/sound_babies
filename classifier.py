#Classify the 2 second audio files based on the wav audio files using a trained model

import os
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from scipy.io import wavfile
import numpy as np

# Load model
extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def classify_audio(audio_data):
    # Convert audio data to torch tensor
    audio_tensor = torch.tensor(audio_data)
    
    # Extract features
    inputs = extractor.feature_extractor(audio_tensor, return_tensors="pt")
    
    # Perform inference
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    # Get predicted labels
    predicted_labels = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    return predicted_labels

# Example usage
# Assuming audio_data is your preprocessed audio data

# Load audio file
sample_rate, audio_data = wavfile.read('240527 Subject 3_segment_58_compressed.wav')

predicted_labels = classify_audio(audio_data)
print(predicted_labels)

