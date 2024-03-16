#This piece of code should form a model of what each sound type is, based on the parent folder in the Control Kit, and then analyze a sound file that the user provides and determine what type of sound it is.
#This will then be extended, upon satisfactory success rate, to automatically sorting a list of sounds rather than just a singular one.

import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
from tkinter import filedialog
from tkinter import *
from audioread import exceptions as arErrors

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

# Function to load the dataset and extract labeled features
def load_dataset(base_path):
    features, labels = [], []
    label_names = os.listdir(base_path)
    label_dict = {name: i for i, name in enumerate(label_names)}
    max_length = 0  # Initialize a variable to keep track of the maximum feature length

    for label_name in label_names:
        label_folder = os.path.join(base_path, label_name)
        for file in os.listdir(label_folder):
            if file.endswith('.mp3') or file.endswith('.wav'):
                file_path = os.path.join(label_folder, file)
                try:
                    file_path = os.path.join(label_folder, file)
                    extracted_features = extract_features(file_path)
                    flattened_features = np.concatenate([feature.flatten() for feature in extracted_features])
                    max_length = max(max_length, flattened_features.shape[0])  # Update max_length if this feature vector is longer
                    features.append(flattened_features)
                    labels.append(label_dict[label_name])
                    print(f'{file_path} successfully added to training data :D')
                except:
                    print(f'{file_path} is not supported by this program :(')
    
    # Now max_length contains the maximum feature vector length
    # Pad features to have uniform length
    features = [np.pad(feature, (0, max_length - len(feature)), 'constant', constant_values=0) for feature in features]

    return np.array(features), np.array(labels), label_names, max_length



# Train a machine learning model to learn from the Control Kit
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Multi-layer Perceptron classifier
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), solver='lbfgs', activation='relu', max_iter=300, verbose=True)
    mlp.fit(X_train, y_train)

    print("Training accuracy: {:.2f}%".format(mlp.score(X_train, y_train) * 100))
    print("Test accuracy: {:.2f}%".format(mlp.score(X_test, y_test) * 100))

    # Save the trained model and the scaler
    joblib.dump(mlp, 'sound_classifier.model')
    joblib.dump(scaler, 'scaler.model')
    
    return mlp, scaler

# Function to predict the class of new sounds
def classify_sound(model, scaler, audio_file):
    # Load the maximum feature length
    with open('max_feature_length.txt', 'r') as f:
        max_feature_length = int(f.read())

    features = extract_features(audio_file)
    flattened_features = np.concatenate([feature.flatten() for feature in features])

    # Pad the feature vector to have the same length as the training feature vectors
    if len(flattened_features) < max_feature_length:
        flattened_features = np.pad(flattened_features, (0, max_feature_length - len(flattened_features)), 'constant')
    elif len(flattened_features) > max_feature_length:
        flattened_features = flattened_features[:max_feature_length]

    normalized_features = scaler.transform([flattened_features])
    prediction = model.predict(normalized_features)
    return prediction


# GUI to select folder path
def get_folder_path():
    root = Tk()
    root.withdraw() # Hide the main window
    folder_path = filedialog.askdirectory()
    return folder_path

# GUI to select folder path
# GUI to select file path
def get_file_path():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Changed from askopenfile to askopenfilename
    return file_path

# Main function to load data, train model, and classify new sounds
def main():
    # Get the full path of the current script
    script_path = os.path.abspath(__file__)

    # Extract the directory path where the current script is located
    script_directory = os.path.dirname(script_path)

    control_folder = script_directory + "\\Control Kit"
    print(control_folder)
    new_sound_path = get_file_path()

    # Load the dataset
    features, labels, label_names, max_feature_length = load_dataset(control_folder)

    # Save the maximum feature length to a file
    with open('max_feature_length.txt', 'w') as f:
        f.write(str(max_feature_length))
    
    # Train the model
    mlp, scaler = train_model(features, labels)
    
    # Classify a new sound
    predicted_label = classify_sound(mlp, scaler, new_sound_path)
    print(f"The sound is a: {label_names[predicted_label[0]]}")

if __name__ == "__main__":
    main()
