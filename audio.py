import os
import librosa
import numpy as np
import soundfile as sf
import csv
from pydub import AudioSegment

def convert_to_wav(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mp3') or file_name.endswith('.m4a'):
            print(f"Converting {file_name} to WAV...")
            file_path = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.wav")
            audio = AudioSegment.from_file(file_path)
            audio.export(output_file, format="wav")
            print(f"Converted {file_name} to WAV.")

def segment_audio(file_name, segment_length=2.0):
    audio, sample_rate = librosa.load(file_name, sr=None)
    segments = []
    segment_samples = int(segment_length * sample_rate)
    for start in range(0, len(audio), segment_samples):
        end = start + segment_samples
        segment = audio[start:end]
        if len(segment) == segment_samples:
            segments.append(segment)
    return segments, sample_rate

def extract_features(segment, sample_rate):
    mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=40)
    rms = librosa.feature.rms(y=segment)
    return np.hstack((np.mean(mfccs.T, axis=0), np.mean(rms.T, axis=0)))

def is_silent(segment, sample_rate, silence_threshold=0.01):
    rms = librosa.feature.rms(y=segment)[0]
    return np.mean(rms) < silence_threshold

def load_and_segment_data(data_path, output_path, csv_path, segment_length=2.0, silence_threshold=0.001):
    segments_features = []
    segments_audio = []
    os.makedirs(output_path, exist_ok=True)
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Segment', 'Start Time', 'End Time', 'Classification'])  # Write CSV header
        for file_name in os.listdir(data_path):
            if file_name.endswith('.wav'):  # Ensure only WAV audio files are processed
                file_path = os.path.join(data_path, file_name)
                segments, sample_rate = segment_audio(file_path, segment_length)
                for i, segment in enumerate(segments):
                    start_time = i * segment_length
                    end_time = start_time + segment_length
                    segment_name = f"{os.path.splitext(file_name)[0]}_segment_{i}.wav"
                    if not is_silent(segment, sample_rate, silence_threshold):
                        features = extract_features(segment, sample_rate)
                        segments_features.append(features)
                        segments_audio.append(segment)
                        # Save the segment to the output folder
                        output_file = os.path.join(output_path, segment_name)
                        sf.write(output_file, segment, sample_rate)
                        writer.writerow([segment_name, start_time, end_time, 'not silent'])  # Write to CSV
                    else:
                        writer.writerow([segment_name, start_time, end_time, 'silent'])  # Write to CSV
    return np.array(segments_features), segments_audio

input_folder = 'audio/'  # Path to the input audio files folder
wav_folder = 'audio_wav/'  # Temporary folder to save converted WAV files
output_folder = 'output_segments/'  # Path to save the segments
csv_path = 'segments_classification.csv'  # Path to save the CSV file

# Convert MP3 and M4A files to WAV
convert_to_wav(input_folder, wav_folder)

# Load, segment, and classify the WAV files
segments_features, segments_audio = load_and_segment_data(wav_folder, output_folder, csv_path)
