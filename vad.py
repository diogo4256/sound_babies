import librosa
import numpy as np
import soundfile as sf


def rms_energy(audio, frame_length=2048, hop_length=512):
    """Calculate RMS energy of audio."""
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return rms

def split_on_rms(audio, min_rms, min_silence=100):
    """Split audio based on RMS energy."""
    chunks = []
    start_idx = 0
    chunk_start = None
    frame_length = 2048
    hop_length = 512
    
    rms_values = rms_energy(audio, frame_length=frame_length, hop_length=hop_length)
    
    for i, rms_value in enumerate(rms_values):
        if rms_value > min_rms:
            if chunk_start is None:
                chunk_start = i * hop_length
        else:
            if chunk_start is not None:
                if (i * hop_length - chunk_start) >= min_silence:
                    chunks.append(audio[start_idx:i * hop_length])
                    start_idx = i * hop_length
                    chunk_start = None
    
    # Add the last chunk if it meets the condition
    if chunk_start is not None:
        chunks.append(audio[start_idx:])
    
    return chunks

# Load audio file
audio_file = "audio/240516 Subject 4-1.wav"
audio, sr = librosa.load(audio_file)

# Split audio based on RMS energy
min_rms_threshold = 0.002  # Adjust as needed
min_silence_duration = 500  # Adjust as needed
chunks = split_on_rms(audio, min_rms=min_rms_threshold, min_silence=min_silence_duration)

# Export non-silent parts
for i, chunk in enumerate(chunks):
    #librosa.output.write_wav(f"vad_output/output_{i}.wav", chunk, sr)

    sf.write(f'vad_output/file{i}.wav', chunk, sr, 'PCM_16')
