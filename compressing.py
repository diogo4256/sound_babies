import os
from pydub import AudioSegment
import librosa
import soundfile as sf

def resample_audio(input_file, output_file, target_sample_rate=16000, target_bit_depth=16):
    # Load audio using librosa
    audio, sample_rate = librosa.load(input_file, sr=None)
    
    # Resample if necessary
    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr= sample_rate, target_sr= target_sample_rate)
    
    # Convert to mono
    audio = librosa.to_mono(audio)
    
    # Write the resampled audio to a temporary file
    temp_file = 'temp_audio.wav'
    sf.write(temp_file, audio, target_sample_rate, subtype='PCM_' + str(target_bit_depth))
    
    # Load the temp file with pydub and export with desired bit depth
    audio_segment = AudioSegment.from_wav(temp_file)
    audio_segment.export(output_file, format="wav", bitrate=f"{target_bit_depth}k")

def process_all_files_in_folder(input_folder, output_folder, target_sample_rate=16000, target_bit_depth=16):
    os.makedirs(output_folder, exist_ok=True)  # Create output directory if it doesn't exist
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_compressed.wav")
            resample_audio(input_file, output_file, target_sample_rate, target_bit_depth)
            print(f"Processed {file_name}")


target_sample_rate = 16000  # Desired sample rate (e.g., 16 kHz)
target_bit_depth = 16  # Desired bit depth (e.g., 16-bit)
input_folder = 'output_segments/'  # Path to the input audio files folder
output_folder = f'output_segments_compressed_{target_sample_rate}/'  # Path to the output folder for compressed files

process_all_files_in_folder(input_folder, output_folder, target_sample_rate, target_bit_depth)

