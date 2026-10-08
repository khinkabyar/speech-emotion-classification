import os
import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================================================
# SETTINGS
# =========================================================
DATASET_PATH = r"D:\SER\RAVDESS"
TARGET_SR = 22050
DURATION = 3
TARGET_LENGTH = TARGET_SR * DURATION

PRE_EMPHASIS = 0.97

# TARGET_SR = 22050
# TARGET_DURATION = 3.0  # Exact 3 seconds
# TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)  # 22050 * 3 = 66150 samples

# =========================================================
# Pre-emphasis
# =========================================================
def apply_pre_emphasis(data, alpha=PRE_EMPHASIS):

    """
    Pre-emphasis filter:

    y[n] = x[n] - alpha * x[n-1]
    """

    emphasized_data = np.append(
        data[0],
        data[1:] - alpha * data[:-1]
    )

    return emphasized_data


# =========================================================
# FEATURE EXTRACTION
# Total = 360 features (MFCC: 80, Chroma: 24, Mel: 256)
# =========================================================

def extract_features(file_path):
    # 1. Load Audio File (Start ကနေ အပြည့်အဝ ဖတ်မည်)
    data, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    # 2. Audio Length 3 Seconds ရအောင် ညှိခြင်း (Pad or Truncate)
    if len(data) < TARGET_LENGTH:
        # ၃ စက္ကန့်ထက် တိုပါက အနောက်တွင် 0 ဖြည့်မည်
        data = np.pad(data, (0, TARGET_LENGTH - len(data)), mode="constant")
    else:
        # ၃ စက္ကန့်ထက် ရှည်ပါက ၃ စက္ကန့်အထိ ဖြတ်ယူမည်
        data = data[:TARGET_LENGTH]
    # -----------------------------------------------------
        # Pre-emphasis
        # -----------------------------------------------------
    
        data = apply_pre_emphasis(data)
    

    # 3. MFCC - 40 (Mean: 40, Std: 40)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 4. Chroma - 12 (Mean: 12, Std: 12)
    stft = np.abs(librosa.stft(data))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # 5. Mel Spectrogram - 128 (Mean: 128, Std: 128)
    mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
    mel_mean = np.mean(mel, axis=1)
    mel_std = np.std(mel, axis=1)

    # Feature Matrix ပေါင်းစပ်ခြင်း (80 + 24 + 256 = 360)
    features = np.hstack(
        [mfcc_mean, mfcc_std, chroma_mean, chroma_std, mel_mean, mel_std]
    )
    return features

# =========================================================
# RAVDESS EMOTION LABELS
# =========================================================
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# =========================================================
# LOAD DATASET
# =========================================================
X = []
y = []


print("Loading RAVDESS Dataset...")
print("Pre-emphasis:", PRE_EMPHASIS)
print()
print("======================================")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path မတွေ့ပါ: {DATASET_PATH}")

for actor in sorted(os.listdir(DATASET_PATH)):
    actor_path = os.path.join(DATASET_PATH, actor)
    if not os.path.isdir(actor_path):
        continue

    for file in sorted(os.listdir(actor_path)):
        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)
        parts = file.split("-")
        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        if emotion_code not in emotion_dict:
            continue

        emotion = emotion_dict[emotion_code]

        try:
            features = extract_features(file_path)

            if len(features) != 360:
                print(
                    f"Wrong feature length: {file_path} (Got {len(features)})"
                )
                continue

            X.append(features)
            y.append(emotion)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# =========================================================
# PREPARE DATA
# =========================================================
X = np.array(X)
y = np.array(y)

print("\n======================================")
print("Dataset Loaded")
print("======================================")
print("X shape:", X.shape)
print("Number of samples:", len(X))
print("Number of features:", X.shape[1] if len(X) > 0 else 0)

print("\nClass Distribution:")
for emotion in sorted(emotion_dict.values()):
    count = np.sum(y == emotion)
    print(f"{emotion:10s}: {count}")

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\n======================================")
print("Emotion Classes")
print("======================================")
print(le.classes_)

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# MLP CLASSIFIER
# =========================================================
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate_init=0.0005,
    learning_rate="adaptive",
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    batch_size=32,
    random_state=42,
)

# =========================================================
# TRAIN
# =========================================================
print("\n======================================")
print("Training MLP Model...")
print("======================================")
mlp.fit(X_train_scaled, y_train)
print("Training completed!")

# =========================================================
# PREDICTION & EVALUATION
# =========================================================
y_train_pred = mlp.predict(X_train_scaled)
y_test_pred = mlp.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n======================================")
print("Accuracy")
print("======================================")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy : {test_accuracy * 100:.2f}%")

print("\n======================================")
print("Classification Report")
print("======================================")
print(
    classification_report(y_test, y_test_pred, target_names=le.classes_)
)

# Confusion Matrix Visualisation
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap="Blues",
)
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# =========================================================
# SAVE MODEL & ARTIFACTS
# =========================================================
joblib.dump(mlp, "mlp_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n======================================")
print("MODEL FILES SAVED SUCCESSFULLY")
print("======================================")
print("mlp_model.pkl")
print("scaler.pkl")
print("label_encoder.pkl")
print("\nTraining Finished Successfully!")