from transformers import AutoProcessor, ASTModel
import torch
from scipy.io import wavfile
import matplotlib.pyplot as plt


# Load model
processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def classify_audio(audio_file_path):
    # Load audio file
    sample_rate, audio_data = wavfile.read(audio_file_path)

    # Process the audio file
    inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the last hidden state
    last_hidden_state = outputs.last_hidden_state.squeeze().cpu().numpy()

    return last_hidden_state

# Example usage
last_hidden_state = classify_audio('output_segments_compressed_16000/240527 Subject 3_segment_58_compressed.wav')

# Plot the last hidden state
plt.figure(figsize=(10, 5))
plt.plot(last_hidden_state)
plt.title('Last Hidden State')
plt.show()