from transformers import ASTFeatureExtractor
import torch
import torchaudio
from transformers import AutoModelForAudioClassification
import os
import random

# Load the pretrained model
feature_extractor = ASTFeatureExtractor()
model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# Directory containing the audio files
directory = "output_segments_compressed_16000/"

# Initialize an empty pool of unlabeled samples
unlabeled_samples = []

# Active learning loop (repeat as needed)
for _ in range(10):  # You can adjust the number of iterations
    # Select an uncertain sample (randomly for demonstration purposes)
    selected_filename = random.choice(os.listdir(directory))
    selected_waveform, sampling_rate = torchaudio.load(directory + selected_filename)
    selected_waveform = selected_waveform.squeeze().numpy()

    # Process the audio file
    inputs = feature_extractor(selected_waveform, sampling_rate=sampling_rate, padding="max_length", return_tensors="pt")
    input_values = inputs.input_values

    # Perform inference
    with torch.no_grad():
        outputs = model(input_values)

    # Check if the sample is uncertain (you can define your own threshold)
    uncertainty_threshold = 0.7
    if max(outputs.logits[0]) < uncertainty_threshold:
        # Prompt user for correct label
        user_label = input(f"Enter the correct label for '{directory}{selected_filename}': ")
        unlabeled_samples.append((selected_filename, selected_waveform, user_label))

# Now you can annotate the uncertain samples and fine-tune the model
# Repeat the loop with newly labeled samples

# Note: In practice, you'd replace random selection with a more sophisticated strategy.

print(f"Total unlabeled samples: {len(unlabeled_samples)}")
