from transformers import ASTFeatureExtractor
import torch
import torchaudio
from transformers import AutoModelForAudioClassification
import os

feature_extractor = ASTFeatureExtractor()
model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")


# Directory containing the audio files for subject 3
directory = "output_segments_compressed_16000/"

# Loop over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is for subject 3
    if "Subject 3" in filename:
        # Load the audio file
        waveform, sampling_rate = torchaudio.load(directory + filename)
        waveform = waveform.squeeze().numpy()

        # Process the audio file
        inputs = feature_extractor(waveform, sampling_rate=sampling_rate, padding="max_length", return_tensors="pt")
        input_values = inputs.input_values

        # Perform inference
        with torch.no_grad():
            outputs = model(input_values)

        # Get the predicted class
        predicted_class_idx = outputs.logits.argmax(-1).item()
        print(f"Predicted class for {filename}: {model.config.id2label[predicted_class_idx]}")
        