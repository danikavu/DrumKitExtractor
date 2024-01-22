import argparse
import os
import re
import subprocess
from typing import List

import aubio
import librosa
import numpy as np
from pydub import AudioSegment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import soundfile as sf


HOMEPATH = os.path.expanduser("~")


def split_filepath(filepath):
    # Extract the filename from the filepath
    filename = os.path.basename(filepath)
    parts = filename.split('.')
    result = "_".join(parts[:-1])

    return result


def extract_samples(filepath):
    # Step 1: Create an Aubio onset detection object
    onset_detector = aubio.onset("energy")

    # Step 2: Open an audio file (replace with your audio file path)
    audio_file = os.path.join(filepath, "drums.wav")

    # Specify the directory path you want to create
    directory_path = os.path.join(f"{os.sep}".join(audio_file.split(f"{os.sep}")[:-1]), "split_samples")
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create it
        os.makedirs(directory_path)

    # Read the audio file using soundfile
    audio_data, sample_rate = sf.read(audio_file)

    # Step 3: Convert to mono if the audio is stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Step 3: Convert audio data to float32 if needed
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Step 4: Process the audio and detect onsets (drum peaks)
    onset_positions = []

    # Process audio in chunks
    chunk_size = 512  # Set the buffer size to match the onset detector
    for i in range(0, len(audio_data) - chunk_size, chunk_size):
        samples = audio_data[i:i + chunk_size]
        onset = onset_detector(samples)

        if onset:
            onset_positions.append(int(i + onset[0]))  # Convert to integer

    # Step 5: Slice the audio based on onset positions
    sliced_audio = []
    prev_position = 0

    for position in onset_positions:
        n = 2200
        if prev_position > n:
            segment = audio_data[prev_position - n:position - n]
        else:
            segment = audio_data[prev_position:position]
        sliced_audio.append(segment)
        prev_position = position

    # Add the last segment
    sliced_audio.append(audio_data[prev_position:])

    # Step 6: Save the segments as separate audio files
    num_samples = 0
    for i, segment in enumerate(sliced_audio):
        output_file = os.path.join(directory_path, f"segment_{i}.wav")
        sf.write(output_file, segment, sample_rate)
        num_samples = i
    print(f"Extracted {num_samples} samples")

    # Step 1: Specify the folder containing the audio segments
    segment_folder = directory_path

    # Step 2: Create a list to store audio data and filenames
    audio_segments = []
    segment_filenames = []

    # Step 3: Loop through the WAV files in the folder and read them
    for filename in os.listdir(segment_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(segment_folder, filename)
            audio_data, sample_rate = sf.read(file_path)
            audio_segments.append(audio_data)
            segment_filenames.append(filename)

    # Step 4: Convert audio data to float32 if needed
    audio_segments_float32 = [segment.astype(np.float32) for segment in audio_segments]

    # Step 5: Extract MFCC features from the audio segments using librosa
    def extract_mfcc(audio_data, sample_rate):
        mfcc_features = []

        for segment in audio_data:
            mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
            # Pad or truncate the MFCC features to a fixed length
            if mfcc.shape[1] < 100:  # Adjust the desired length as needed
                mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :100]
            mfcc_features.append(mfcc)

        return np.array(mfcc_features)

    # Step 6: Extract MFCC features from the audio segments
    mfcc_features = extract_mfcc(audio_segments_float32, sample_rate)

    # Step 7: Normalize features
    normalized_features = StandardScaler().fit_transform(mfcc_features.reshape(mfcc_features.shape[0], -1))

    # Step 8: Apply K-Means clustering with a fixed number of clusters
    num_clusters = 24  # You can adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_features)

    # Step 9: Organize segments based on cluster labels
    def organize_segments(filenames, cluster_labels):
        clusters = {}
        for filename, label in zip(filenames, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(filename)
        return clusters

    # Step 10: Organize segments based on cluster labels
    segment_clusters = organize_segments(segment_filenames, cluster_labels)
    print(f"Clustering finished. Created {len(segment_clusters)} clusters")

    def rename(directory, filename, group):
        old_name = os.path.join(directory, filename)
        new_name = os.path.join(directory, f"_{group}_{filename}")
        os.rename(old_name, new_name)

    for group in segment_clusters:
        if len(segment_clusters[group]) > 1:
            [rename(directory_path, x, group) for x in segment_clusters[group]]

    _clusters = list(set(re.findall(r"_\d+_", str(os.listdir(directory_path)))))

    for clus in _clusters:
        segs = [x for x in os.listdir(directory_path) if clus in x]
        sound1 = AudioSegment.from_file(os.path.join(directory_path, segs[0]), format="wav")
        for file in segs[1:]:
            filepath = directory_path + f"{os.sep}{file}"
            sound2 = AudioSegment.from_file(filepath, format="wav")
            overlay = sound2.overlay(sound1, position=0)
            os.remove(filepath)
        sf.write(os.path.join(directory_path, f"_{clus}stacked.wav"), overlay.get_array_of_samples(), sample_rate)
        os.remove(os.path.join(directory_path, segs[0]))
    print(f"Drum samples extracted at: {directory_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract drum samples from audio track')
    parser.add_argument('filepath', type=str, help='Filepath of audio file')
    parser.add_argument('--model', type=str, default='mdx_extra', help='Model to be used for extraction')

    args = parser.parse_args()

    print("Filepath of input audio:", args.filepath)
    print("Track separator model used:", args.model)

    # Construct the demucs command based on the arguments
    demucs_command = f'demucs "{args.filepath}" --two-stems=drums -d cuda -n {args.model}'

    # Execute the demucs command
    try:
        subprocess.run(demucs_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing demucs: {e}")
    exported_filepath = os.path.join(HOMEPATH, 'separated', args.model, split_filepath(args.filepath))
    print(f"Separated track exported at: {exported_filepath}")
    extract_samples(exported_filepath)


if __name__ == "__main__":
    main()
