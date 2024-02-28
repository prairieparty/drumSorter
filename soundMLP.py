import os
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
import keras

# Function to extract MFCC, Spectral Centroid, Chroma Features, and Zero Crossing Rate
def extract_features(audio_file, sr=22050, n_mfcc=13, hop_length=512):
    y, sr = librosa.load(audio_file, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    return mfccs, centroid, chroma, zero_crossing_rate

# Function to normalize features globally
def normalize_features(features_list):
    # Flatten and combine the same type of features across all files for normalization
    all_mfccs = np.hstack([features[0] for features in features_list])
    all_centroids = np.hstack([features[1] for features in features_list])
    all_chromas = np.hstack([features[2] for features in features_list])
    all_zcrs = np.hstack([features[3] for features in features_list])
    
    # Normalize each feature type
    scaler = StandardScaler()
    norm_mfccs = scaler.fit_transform(all_mfccs.T).T
    norm_centroids = scaler.fit_transform(all_centroids.T).T
    norm_chromas = scaler.fit_transform(all_chromas.T).T
    norm_zcrs = scaler.fit_transform(all_zcrs.T).T
    
    # Reassemble the features list with normalized features
    normalized_features = []
    for i in range(len(features_list)):
        normalized_features.append((
            norm_mfccs[:, i * norm_mfccs.shape[1] // len(features_list):(i + 1) * norm_mfccs.shape[1] // len(features_list)],
            norm_centroids[:, i * norm_centroids.shape[1] // len(features_list):(i + 1) * norm_centroids.shape[1] // len(features_list)],
            norm_chromas[:, i * norm_chromas.shape[1] // len(features_list):(i + 1) * norm_chromas.shape[1] // len(features_list)],
            norm_zcrs[:, i * norm_zcrs.shape[1] // len(features_list):(i + 1) * norm_zcrs.shape[1] // len(features_list)]
        ))
    return normalized_features

# Function to calculate similarity matrix directly from normalized features
def calculate_similarity_matrix(features_list):
    num_files = len(features_list)
    similarity_matrix = np.zeros((num_files, num_files))
    
    # Preprocess features for similarity calculation
    processed_features = []
    for features in features_list:
        # Flatten each feature set
        flattened_features = np.concatenate([feature.flatten() for feature in features])
        processed_features.append(flattened_features)
    
    # Ensure all feature vectors are of the same length
    max_length = max(len(features) for features in processed_features)
    uniform_features = [np.pad(features, (0, max_length - len(features)), 'constant') for features in processed_features]

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(uniform_features)

    return similarity_matrix


# Function to plot similarity matrix
def plot_similarity_matrix(similarity_matrix, file_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis', origin='lower', interpolation='nearest')
    plt.colorbar(label='Similarity')
    plt.title('Similarity Matrix of Audio Files')
    plt.xlabel('Audio Files')
    plt.ylabel('Audio Files')
    plt.xticks(range(len(file_names)), file_names, rotation=90)
    plt.yticks(range(len(file_names)), file_names)
    plt.tight_layout()
    plt.show()

# Function to load dataset (both control samples and test samples)
def load_dataset(folder_path, control=False):
    labels = []
    features = []
    label_names = ['Kicks', 'Snare', 'Hihat', 'Cymbal', 'Tom', 'Clicks']
    label_dict = {name: i for i, name in enumerate(label_names)}
    if control:
        # Load control samples
        for label_name in label_names:
            label_folder = os.path.join(folder_path, label_name)
            for file in os.listdir(label_folder):
                if file.endswith('.mp3') or file.endswith('.wav'):
                    file_path = os.path.join(label_folder, file)
                    features.append(extract_features(file_path))
                    labels.append(label_dict[label_name])
        return np.array(features), np.array(labels)
    else:
        # Load test samples (Assuming no labels, just features)
        test_features = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.mp3') or file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    test_features.append(extract_features(file_path))
        return np.array(test_features)
    
# Main function to gather features and plot similarity matrix
def matrixCalc(folder_path):
    features_list = []
    file_names = []

    # Iterate through all files in the directory and subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mp3') or file.endswith('.wav'):
                file_path = os.path.join(root, file)
                features = extract_features(file_path)
                features_list.append(features)
                file_names.append(file)

    # Normalize features globally
    normalized_features = normalize_features(features_list)

    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(normalized_features)

    return normalize_features, similarity_matrix

# GUI to select folder path
def get_folder_path():
    root = Tk()
    root.withdraw() # Hide the main window
    folder_path = filedialog.askdirectory()
    return folder_path

def machine_learn(features):
    model = keras.models.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(features.shape[1],)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
def main():
    control_folder = "" #sample pack to test against
    folder_path = get_folder_path()
    if folder_path: # Ensure a folder path was selected
        matrixCalc(folder_path)
        matrixCalc(control_folder)

    else:
        print("No folder selected. Exiting...")
        exit()

if __name__ == "__main__":
    main()
