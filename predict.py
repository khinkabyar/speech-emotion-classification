# import os
# import joblib
# import librosa
# import numpy as np

# # =========================================================
# # LOAD MODEL & ARTIFACTS
# # =========================================================
# try:
#     mlp = joblib.load("mlp_model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     le = joblib.load("label_encoder.pkl")
#     print("Model files loaded successfully!")
# except Exception as e:
#     print("Error loading model files:", e)
#     exit()

# # =========================================================
# # SETTINGS (train.py နဲ့ အတိအကျ တူရပါမည်)
# # =========================================================
# TARGET_SR = 22050
# TARGET_DURATION = 3.0  # Exact 3 seconds
# TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)  # 66150 samples


# # =========================================================
# # FEATURE EXTRACTION
# # =========================================================
# def extract_features(file_path):
#     # 1. Load Audio
#     data, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

#     # 2. Audio Length 3 Seconds ရအောင် ညှိခြင်း (Pad or Truncate)
#     if len(data) < TARGET_LENGTH:
#         data = np.pad(data, (0, TARGET_LENGTH - len(data)), mode="constant")
#     else:
#         data = data[:TARGET_LENGTH]

#     # 3. MFCC - 40 (Mean: 40, Std: 40)
#     mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
#     mfcc_mean = np.mean(mfcc, axis=1)
#     mfcc_std = np.std(mfcc, axis=1)

#     # 4. Chroma - 12 (Mean: 12, Std: 12)
#     stft = np.abs(librosa.stft(data))
#     chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
#     chroma_mean = np.mean(chroma, axis=1)
#     chroma_std = np.std(chroma, axis=1)

#     # 5. Mel Spectrogram - 128 (Mean: 128, Std: 128)
#     mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
#     mel_mean = np.mean(mel, axis=1)
#     mel_std = np.std(mel, axis=1)

#     # Combined = 360 features
#     features = np.hstack(
#         [mfcc_mean, mfcc_std, chroma_mean, chroma_std, mel_mean, mel_std]
#     )

#     return features


# # =========================================================
# # PREDICT FUNCTION
# # =========================================================
# def predict_emotion(file_path):
#     features = extract_features(file_path)

#     print("Feature length:", len(features))

#     # Reshape for 1 sample (1, 360)
#     X = features.reshape(1, -1)

#     # Scaling
#     X_scaled = scaler.transform(X)

#     # Predict Emotion Label
#     prediction = mlp.predict(X_scaled)
#     emotion = le.inverse_transform(prediction)[0]

#     # Predict Probabilities
#     probabilities = mlp.predict_proba(X_scaled)[0]
#     confidence = np.max(probabilities) * 100

#     probability_dict = {}
#     for label, probability in zip(le.classes_, probabilities):
#         probability_dict[label] = float(probability)

#     return emotion, confidence, probability_dict


# # =========================================================
# # MAIN EXECUTION
# # =========================================================
# if __name__ == "__main__":
#     print("======================================")
#     print("Speech Emotion Classification")
#     print("======================================")

#     file_path = input("\nEnter audio file path: ").strip().strip("'\"")

#     if not os.path.exists(file_path):
#         print(f"\nError: File မတွေ့ပါ -> {file_path}")
#     else:
#         try:
#             emotion, confidence, probabilities = predict_emotion(file_path)

#             print("\n--------------------------------------")
#             print("Predicted Emotion:", emotion.upper())
#             print("Confidence       :", f"{confidence:.2f}%")

#             print("\nProbabilities:")
#             for emotion_name, probability in sorted(
#                 probabilities.items(), key=lambda x: x[1], reverse=True
#             ):
#                 print(
#                     f"{emotion_name:10s}: {probability * 100:.2f}%"
#                 )

#             print("--------------------------------------")

#         except Exception as e:
#             print("\nError during prediction:", e)

import numpy as np
import librosa
import joblib

# ==========================================
# Settings
# ==========================================
TARGET_SR = 22050
DURATION = 3
TARGET_LENGTH = TARGET_SR * DURATION

PRE_EMPHASIS = 0.97

# ==========================================
# Load trained files
# ==========================================
model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# ==========================================
# Pre-emphasis
# ==========================================
def apply_pre_emphasis(data, alpha=PRE_EMPHASIS):
    """
    Apply pre-emphasis filter to speech signal.
    y[n] = x[n] - alpha * x[n-1]
    """
    emphasized_data = np.append(
        data[0],
        data[1:] - alpha * data[:-1]
    )

    return emphasized_data


# ==========================================
# Feature Extraction
# ==========================================
def extract_features(file_path):

    # Load audio
    data, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )

    # ======================================
    # Pad / Truncate to exactly 3 seconds
    # ======================================
    if len(data) < TARGET_LENGTH:
        data = np.pad(
            data,
            (0, TARGET_LENGTH - len(data)),
            mode="constant"
        )
    else:
        data = data[:TARGET_LENGTH]

    # ======================================
    # Pre-emphasis
    # ======================================
    data = apply_pre_emphasis(data)

    # ======================================
    # MFCC - 40
    # ======================================
    mfcc = librosa.feature.mfcc(
        y=data,
        sr=sr,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # ======================================
    # Chroma - 12
    # ======================================
    chroma = librosa.feature.chroma_stft(
        y=data,
        sr=sr
    )

    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # ======================================
    # Mel Spectrogram - 128
    # ======================================
    mel = librosa.feature.melspectrogram(
        y=data,
        sr=sr,
        n_mels=128
    )

    mel_mean = np.mean(mel, axis=1)
    mel_std = np.std(mel, axis=1)

    # ======================================
    # Combine features
    # ======================================
    features = np.hstack([
        mfcc_mean,
        mfcc_std,
        chroma_mean,
        chroma_std,
        mel_mean,
        mel_std
    ])

    return features


# ==========================================
# Prediction
# ==========================================
def predict_emotion(file_path):

    # Extract 360 features
    features = extract_features(file_path)

    print("Number of features:", len(features))

    # Reshape for model
    features = features.reshape(1, -1)

    # Standardization
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)

    # Emotion label
    emotion = label_encoder.inverse_transform(prediction)[0]

    # Probability
    probabilities = model.predict_proba(features_scaled)[0]

    # Sort probabilities from high to low
    results = sorted(
        zip(label_encoder.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n==============================")
    print("Speech Emotion Prediction")
    print("==============================")
    print("Predicted Emotion:", emotion)
    print("\nEmotion Probabilities:")

    for label, probability in results:
        print(f"{label:10s}: {probability * 100:.2f}%")

    return emotion, results


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":

    file_path = input("Enter audio file path: ").strip()

    predict_emotion(file_path)

