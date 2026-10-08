import os
import joblib
import librosa
import numpy as np

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


# =========================================================
# Flask App
# =========================================================

app = Flask(__name__)


# =========================================================
# Upload Settings
# =========================================================

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    "wav",
    "mp3",
    "m4a",
    "flac",
    "ogg"
}

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# Load Model and Preprocessing Artifacts
# =========================================================

try:

    mlp = joblib.load(
        "mlp_model.pkl"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    le = joblib.load(
        "label_encoder.pkl"
    )

    print(
        "Model, Scaler, and LabelEncoder loaded successfully!"
    )

except Exception as e:

    print(
        f"Error loading model files: {e}"
    )

    mlp = None
    scaler = None
    le = None


# =========================================================
# Audio Settings
# =========================================================

# IMPORTANT:
# This must match the settings used during training.

TARGET_SR = 22050

TARGET_DURATION = 3.0

TARGET_LENGTH = int(
    TARGET_SR * TARGET_DURATION
)


# =========================================================
# Pre-emphasis
# =========================================================

PRE_EMPHASIS = 0.97


# =========================================================
# Allowed File
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# Pre-emphasis Function
# =========================================================

def apply_pre_emphasis(
    data,
    alpha=PRE_EMPHASIS
):

    """
    Pre-emphasis filter

    y[n] = x[n] - alpha * x[n-1]
    """

    if len(data) == 0:
        return data

    emphasized_data = np.append(
        data[0],
        data[1:] -
        alpha * data[:-1]
    )

    return emphasized_data


# =========================================================
# Feature Extraction
# =========================================================

def extract_features(file_path):

    # -----------------------------------------------------
    # 1. Load Audio
    # -----------------------------------------------------

    data, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )


    # -----------------------------------------------------
    # 2. Check Empty Audio
    # -----------------------------------------------------

    if data is None or len(data) == 0:

        raise ValueError(
            "The audio file contains no audio data."
        )


    # -----------------------------------------------------
    # 3. Length Adjustment
    #
    # Exact TARGET_DURATION seconds
    # -----------------------------------------------------

    if len(data) < TARGET_LENGTH:

        data = np.pad(
            data,
            (
                0,
                TARGET_LENGTH - len(data)
            ),
            mode="constant"
        )

    else:

        data = data[:TARGET_LENGTH]


    # -----------------------------------------------------
    # 4. Original Waveform
    # -----------------------------------------------------

    original_waveform = data.copy()


    # -----------------------------------------------------
    # 5. Pre-emphasis
    # -----------------------------------------------------

    emphasized_data = apply_pre_emphasis(
        data,
        PRE_EMPHASIS
    )


    # =====================================================
    # 6. MFCC
    #
    # 40 coefficients
    # Mean = 40
    # Std  = 40
    # Total = 80
    # =====================================================

    mfcc = librosa.feature.mfcc(
        y=emphasized_data,
        sr=sr,
        n_mfcc=40
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )


    # =====================================================
    # 7. Chroma
    #
    # 12 coefficients
    # Mean = 12
    # Std  = 12
    # Total = 24
    # =====================================================

    stft = np.abs(
        librosa.stft(
            emphasized_data
        )
    )

    chroma = librosa.feature.chroma_stft(
        S=stft,
        sr=sr
    )

    chroma_mean = np.mean(
        chroma,
        axis=1
    )

    chroma_std = np.std(
        chroma,
        axis=1
    )


    # =====================================================
    # 8. Mel-Spectrogram
    #
    # 128 Mel bands
    # Mean = 128
    # Std  = 128
    # Total = 256
    # =====================================================

    mel = librosa.feature.melspectrogram(
        y=emphasized_data,
        sr=sr,
        n_mels=128
    )

    mel_mean = np.mean(
        mel,
        axis=1
    )

    mel_std = np.std(
        mel,
        axis=1
    )


    # =====================================================
    # 9. Combine Features
    #
    # MFCC   = 80
    # Chroma = 24
    # Mel    = 256
    #
    # Total  = 360
    # =====================================================

    combined_features = np.hstack([
        mfcc_mean,
        mfcc_std,
        chroma_mean,
        chroma_std,
        mel_mean,
        mel_std
    ])


    # -----------------------------------------------------
    # Check Feature Dimension
    # -----------------------------------------------------

    if len(combined_features) != 360:

        raise ValueError(
            f"Expected 360 features, "
            f"but extracted "
            f"{len(combined_features)} features."
        )


    # =====================================================
    # 10. Raw Features for GUI
    # =====================================================

    raw_features = {}


    # -----------------------------------------------------
    # MFCC Mean
    # -----------------------------------------------------

    for i in range(40):

        raw_features[
            f"MFCC_{i + 1}_mean"
        ] = float(
            mfcc_mean[i]
        )


    # -----------------------------------------------------
    # MFCC Std
    # -----------------------------------------------------

    for i in range(40):

        raw_features[
            f"MFCC_{i + 1}_std"
        ] = float(
            mfcc_std[i]
        )


    # -----------------------------------------------------
    # Chroma Mean
    # -----------------------------------------------------

    for i in range(12):

        raw_features[
            f"Chroma_{i + 1}_mean"
        ] = float(
            chroma_mean[i]
        )


    # -----------------------------------------------------
    # Chroma Std
    # -----------------------------------------------------

    for i in range(12):

        raw_features[
            f"Chroma_{i + 1}_std"
        ] = float(
            chroma_std[i]
        )


    # -----------------------------------------------------
    # Mel Mean
    # -----------------------------------------------------

    for i in range(128):

        raw_features[
            f"Mel_{i + 1}_mean"
        ] = float(
            mel_mean[i]
        )


    # -----------------------------------------------------
    # Mel Std
    # -----------------------------------------------------

    for i in range(128):

        raw_features[
            f"Mel_{i + 1}_std"
        ] = float(
            mel_std[i]
        )


    # =====================================================
    # 11. Waveform Data
    #
    # Used if needed by GUI
    # =====================================================

    max_points = 2000

    step = max(
        1,
        len(original_waveform) //
        max_points
    )


    waveform_original = (
        original_waveform[::step]
        .tolist()
    )


    waveform_emphasized = (
        emphasized_data[::step]
        .tolist()
    )


    waveform_time = (
        np.arange(
            0,
            len(original_waveform),
            step
        ) / sr
    ).tolist()


    # =====================================================
    # 12. Return
    # =====================================================

    return (
        combined_features,
        raw_features,
        waveform_original,
        waveform_emphasized,
        waveform_time,
        sr
    )


# =========================================================
# Home Page
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# EXTRACT FEATURES
# =========================================================

@app.route(
    "/extract_features",
    methods=["POST"]
)
def extract_features_route():

    # -----------------------------------------------------
    # Check audio file
    # -----------------------------------------------------

    if "audio" not in request.files:

        return jsonify({

            "success": False,

            "error":
                "No audio file provided."

        }), 400


    file = request.files["audio"]


    # -----------------------------------------------------
    # Check filename
    # -----------------------------------------------------

    if file.filename == "":

        return jsonify({

            "success": False,

            "error":
                "No selected file."

        }), 400


    # -----------------------------------------------------
    # Check extension
    # -----------------------------------------------------

    if not allowed_file(
        file.filename
    ):

        return jsonify({

            "success": False,

            "error":
                "Invalid file format."

        }), 400


    # -----------------------------------------------------
    # Secure filename
    # -----------------------------------------------------

    filename = secure_filename(
        file.filename
    )


    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    try:

        # -------------------------------------------------
        # Save temporary file
        # -------------------------------------------------

        file.save(
            file_path
        )


        # -------------------------------------------------
        # Extract features
        # -------------------------------------------------

        (
            features,
            raw_features,
            waveform_original,
            waveform_emphasized,
            waveform_time,
            sr
        ) = extract_features(
            file_path
        )


        # -------------------------------------------------
        # Convert feature vector to list
        # -------------------------------------------------

        feature_vector = (
            features
            .astype(float)
            .tolist()
        )


        # -------------------------------------------------
        # Remove temporary file
        # -------------------------------------------------

        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )


        # -------------------------------------------------
        # Return feature data
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "filename":
                filename,

            "extracted_features":
                raw_features,

            "feature_vector":
                feature_vector,

            "sample_rate":
                sr,

            "duration":
                TARGET_DURATION,

            "pre_emphasis":
                PRE_EMPHASIS,

            "waveform_original":
                waveform_original,

            "waveform_emphasized":
                waveform_emphasized,

            "waveform_time":
                waveform_time

        }), 200


    except Exception as e:

        # -------------------------------------------------
        # Remove temporary file if error occurs
        # -------------------------------------------------

        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )


        print(
            "Feature extraction error:",
            str(e)
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # -------------------------------------------------
        # Check model
        # -------------------------------------------------

        if (
            mlp is None
            or scaler is None
            or le is None
        ):

            return jsonify({

                "success": False,

                "error":
                    "Model files could not be loaded."

            }), 500


        # -------------------------------------------------
        # Get JSON data
        # -------------------------------------------------

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No JSON data received."

            }), 400


        # -------------------------------------------------
        # Get feature vector
        # -------------------------------------------------

        feature_vector = data.get(
            "feature_vector"
        )


        if feature_vector is None:

            return jsonify({

                "success": False,

                "error":
                    "No feature vector provided."

            }), 400


        # -------------------------------------------------
        # Check feature vector type
        # -----------------------------------------------------

        if not isinstance(
            feature_vector,
            list
        ):

            return jsonify({

                "success": False,

                "error":
                    "Feature vector must be a list."

            }), 400


        # -------------------------------------------------
        # Check feature count
        # -----------------------------------------------------

        if len(feature_vector) != 360:

            return jsonify({

                "success": False,

                "error":
                    (
                        "Expected 360 features, "
                        f"but received "
                        f"{len(feature_vector)}."
                    )

            }), 400


        # -------------------------------------------------
        # Convert to NumPy
        # -------------------------------------------------

        X = np.array(
            feature_vector,
            dtype=np.float64
        ).reshape(
            1,
            -1
        )


        # -------------------------------------------------
        # Check for invalid values
        # -------------------------------------------------

        if not np.all(
            np.isfinite(X)
        ):

            return jsonify({

                "success": False,

                "error":
                    "Feature vector contains invalid values."

            }), 400


        # =================================================
        # Standardization
        # =================================================

        X_scaled = scaler.transform(
            X
        )


        # =================================================
        # MLP Prediction
        # =================================================

        prediction = mlp.predict(
            X_scaled
        )


        emotion = le.inverse_transform(
            prediction
        )[0]


        # =================================================
        # Prediction Probability
        # =================================================

        probabilities = (
            mlp.predict_proba(
                X_scaled
            )[0]
        )


        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        confidence = float(
            np.max(
                probabilities
            ) * 100
        )


        # -------------------------------------------------
        # Probability Dictionary
        # -------------------------------------------------

        probability_dict = {}


        for label, prob in zip(
            le.classes_,
            probabilities
        ):

            probability_dict[
                str(label)
            ] = round(
                float(prob) * 100,
                2
            )


        # =================================================
        # Return Classification Result
        # =================================================

        return jsonify({

            "success": True,

            "emotion":
                str(emotion),

            "confidence":
                round(
                    confidence,
                    2
                ),

            "probabilities":
                probability_dict

        }), 200


    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# =========================================================
# Run Flask
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",

        port=5000,
        debug=True
    )
# import os
# import joblib
# import librosa
# import json
# import numpy as np

# from flask import Flask, jsonify, render_template, request
# from werkzeug.utils import secure_filename


# =========================================================
# Flask App
# =========================================================

# app = Flask(__name__)


# # =========================================================
# # Upload Settings
# # =========================================================

# UPLOAD_FOLDER = "uploads"

# ALLOWED_EXTENSIONS = {
#     "wav",
#     "mp3",
#     "m4a",
#     "flac",
#     "ogg"
# }

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# Load Model & Artifacts
# =========================================================

# try:

#     mlp = joblib.load("mlp_model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     le = joblib.load("label_encoder.pkl")

#     print("Model, Scaler, and LabelEncoder loaded successfully!")

# except Exception as e:

#     print(f"Error loading model files: {e}")


# # =========================================================
# # Audio Settings
# # =========================================================

# TARGET_SR = 22050

# TARGET_DURATION = 3.0

# TARGET_LENGTH = int(
#     TARGET_SR * TARGET_DURATION
# )


# # =========================================================
# # Pre-emphasis Setting
# # =========================================================

# PRE_EMPHASIS = 0.97


# # =========================================================
# # Allowed File
# # =========================================================

# def allowed_file(filename):

#     return (
#         "." in filename
#         and filename.rsplit(".", 1)[1].lower()
#         in ALLOWED_EXTENSIONS
#     )


# # =========================================================
# # Pre-emphasis
# # =========================================================

# def apply_pre_emphasis(data, alpha=PRE_EMPHASIS):

#     """
#     Pre-emphasis filter

#     y[n] = x[n] - alpha * x[n-1]
#     """

#     if len(data) == 0:
#         return data

#     emphasized_data = np.append(
#         data[0],
#         data[1:] - alpha * data[:-1]
#     )

#     return emphasized_data


# =========================================================
# Feature Extraction
# =========================================================

# def extract_features(file_path):

#     # -----------------------------------------------------
#     # 1. Load Audio
#     # -----------------------------------------------------

#     data, sr = librosa.load(
#         file_path,
#         sr=TARGET_SR,
#         mono=True
#     )


#     # -----------------------------------------------------
#     # 2. Length Adjustment
#     #    Exact 3 seconds
#     # -----------------------------------------------------

#     if len(data) < TARGET_LENGTH:

#         data = np.pad(
#             data,
#             (0, TARGET_LENGTH - len(data)),
#             mode="constant"
#         )

#     else:

#         data = data[:TARGET_LENGTH]


#     # -----------------------------------------------------
#     # 3. Keep Original Waveform
#     # -----------------------------------------------------

#     original_waveform = data.copy()


#     # -----------------------------------------------------
#     # 4. Pre-emphasis
#     # -----------------------------------------------------

#     emphasized_data = apply_pre_emphasis(
#         data,
#         PRE_EMPHASIS
#     )


#     # =====================================================
#     # 5. MFCC - 40
#     # =====================================================

#     mfcc = librosa.feature.mfcc(
#         y=emphasized_data,
#         sr=sr,
#         n_mfcc=40
#     )

#     mfcc_mean = np.mean(
#         mfcc,
#         axis=1
#     )

#     mfcc_std = np.std(
#         mfcc,
#         axis=1
#     )


#     # =====================================================
#     # 6. Chroma - 12
#     # =====================================================

#     stft = np.abs(
#         librosa.stft(emphasized_data)
#     )

#     chroma = librosa.feature.chroma_stft(
#         S=stft,
#         sr=sr
#     )

#     chroma_mean = np.mean(
#         chroma,
#         axis=1
#     )

#     chroma_std = np.std(
#         chroma,
#         axis=1
#     )


#     # =====================================================
#     # 7. Mel Spectrogram - 128
#     # =====================================================

#     mel = librosa.feature.melspectrogram(
#         y=emphasized_data,
#         sr=sr,
#         n_mels=128
#     )

#     mel_mean = np.mean(
#         mel,
#         axis=1
#     )

#     mel_std = np.std(
#         mel,
#         axis=1
#     )


#     # =====================================================
#     # 8. Combine = 360 Features
#     # =====================================================

#     combined_features = np.hstack([
#         mfcc_mean,
#         mfcc_std,
#         chroma_mean,
#         chroma_std,
#         mel_mean,
#         mel_std
#     ])
    # =====================================================
    # 9. Raw Features for GUI
    # =====================================================

    # raw_features = {

    #     "mfcc_mean":
    #         mfcc_mean.tolist(),

    #     "mfcc_std":
    #         mfcc_std.tolist(),

    #     "chroma_mean":
    #         chroma_mean.tolist(),

    #     "chroma_std":
    #         chroma_std.tolist(),

    #     "mel_mean":
    #         mel_mean.tolist(),

    #     "mel_std":
    #         mel_std.tolist(),
    # }

# =========================================================
# Feature Extraction
# =========================================================
# ------------------------------------------------
# Get Extracted Feature Vector
# ------------------------------------------------
# def extract_features(file_path):

#     # -----------------------------------------------------
#     # 1. Load Audio
#     # -----------------------------------------------------

#     data, sr = librosa.load(
#         file_path,
#         sr=TARGET_SR,
#         mono=True
#     )


#     # -----------------------------------------------------
#     # 2. Length Adjustment
#     #    Exact 3 seconds
#     # -----------------------------------------------------

#     if len(data) < TARGET_LENGTH:

#         data = np.pad(
#             data,
#             (0, TARGET_LENGTH - len(data)),
#             mode="constant"
#         )

#     else:

#         data = data[:TARGET_LENGTH]


#     # -----------------------------------------------------
#     # 3. Keep Original Waveform
#     # -----------------------------------------------------

#     original_waveform = data.copy()


#     # -----------------------------------------------------
#     # 4. Pre-emphasis
#     # -----------------------------------------------------

#     emphasized_data = apply_pre_emphasis(
#         data,
#         PRE_EMPHASIS
#     )


#     # =====================================================
#     # 5. MFCC - 40
#     # =====================================================

#     mfcc = librosa.feature.mfcc(
#         y=emphasized_data,
#         sr=sr,
#         n_mfcc=40
#     )

#     mfcc_mean = np.mean(
#         mfcc,
#         axis=1
#     )

#     mfcc_std = np.std(
#         mfcc,
#         axis=1
#     )


#     # =====================================================
#     # 6. Chroma - 12
#     # =====================================================

#     stft = np.abs(
#         librosa.stft(emphasized_data)
#     )

#     chroma = librosa.feature.chroma_stft(
#         S=stft,
#         sr=sr
#     )

#     chroma_mean = np.mean(
#         chroma,
#         axis=1
#     )

#     chroma_std = np.std(
#         chroma,
#         axis=1
#     )


#     # =====================================================
#     # 7. Mel Spectrogram - 128
#     # =====================================================

#     mel = librosa.feature.melspectrogram(
#         y=emphasized_data,
#         sr=sr,
#         n_mels=128
#     )

#     mel_mean = np.mean(
#         mel,
#         axis=1
#     )

#     mel_std = np.std(
#         mel,
#         axis=1
#     )


#     # =====================================================
#     # 8. Combine = 360 Features
#     # =====================================================

#     combined_features = np.hstack([
#         mfcc_mean,
#         mfcc_std,
#         chroma_mean,
#         chroma_std,
#         mel_mean,
#         mel_std
#     ])


#     # =====================================================
#     # 9. Raw Features for GUI
#     # =====================================================

#     raw_features = {}

#     for i in range(40):
#         raw_features[f"MFCC_{i + 1}_mean"] = float(mfcc_mean[i])

#     for i in range(40):
#         raw_features[f"MFCC_{i + 1}_std"] = float(mfcc_std[i])

#     for i in range(12):
#         raw_features[f"Chroma_{i + 1}_mean"] = float(chroma_mean[i])

#     for i in range(12):
#         raw_features[f"Chroma_{i + 1}_std"] = float(chroma_std[i])

#     for i in range(128):
#         raw_features[f"Mel_{i + 1}_mean"] = float(mel_mean[i])

#     for i in range(128):
#         raw_features[f"Mel_{i + 1}_std"] = float(mel_std[i])


#     # =====================================================
#     # 10. Waveform Data for GUI
#     # =====================================================

#     max_points = 2000

#     step = max(
#         1,
#         len(original_waveform) // max_points
#     )

#     waveform_original = (
#         original_waveform[::step]
#         .tolist()
#     )

#     waveform_emphasized = (
#         emphasized_data[::step]
#         .tolist()
#     )

    # waveform_time = (
    #     np.arange(
    #         0,
    #         len(original_waveform),
    #         step
    #     ) / sr
    # ).tolist()


    # # =====================================================
    # # RETURN
    # # =====================================================

    # return (
    #     combined_features,
    #     raw_features,
    #     waveform_original,
    #     waveform_emphasized,
    #     waveform_time,
    #     sr
    # )
# # ---------------------------------------------------------
# # Time axis for waveform
# # ---------------------------------------------------------

# waveform_time = (
#     np.arange(
#         0,
#         len(original_waveform),
#         step
#     ) / sr
# ).tolist()


# # =========================================================
# # RETURN
# # =========================================================

# return (
#     combined_features,
#     raw_features,
#     waveform_original,
#     waveform_emphasized,
#     waveform_time,
#     sr
# )


# =========================================================
# Home Page
# =========================================================

# @app.route("/")
# def index():

#     return render_template(
#         "index.html"
#     )


# =========================================================
# Prediction
# =========================================================

# @app.route(
#     "/predict",
#     methods=["POST"]
# )
# def predict():

#     if "audio" not in request.files:

#         return jsonify({
#             "error": "No audio file provided"
#         }), 400


#     file = request.files["audio"]


#     if file.filename == "":

#         return jsonify({
#             "error": "No selected file"
#         }), 400


#     if file and allowed_file(
#         file.filename
#     ):

#         filename = secure_filename(
#             file.filename
#         )

#         file_path = os.path.join(
#             app.config["UPLOAD_FOLDER"],
#             filename
#         )


#         try:

#             # ------------------------------------------------
#             # Save uploaded audio
#             # ------------------------------------------------

#             file.save(file_path)


#             # ------------------------------------------------
#             # Extract Features + Waveform
#             # ------------------------------------------------

#             (
#                 features,
#                 raw_features,
#                 waveform_original,
#                 waveform_emphasized,
#                 waveform_time,
#                 sr
#             ) = extract_features(file_path)


#             # ------------------------------------------------
#             # Reshape
#             # ------------------------------------------------

#             X = features.reshape(
#                 1,
#                 -1
#             )


#             # ------------------------------------------------
#             # Standardization
#             # ------------------------------------------------

#             X_scaled = scaler.transform(
#                 X
#             )


#             # ------------------------------------------------
#             # Prediction
#             # ------------------------------------------------

#             prediction = mlp.predict(
#                 X_scaled
#             )


#             emotion = le.inverse_transform(
#                 prediction
#             )[0]


#             # ------------------------------------------------
#             # Probability
#             # ------------------------------------------------

#             probabilities = mlp.predict_proba(
#                 X_scaled
#             )[0]


#             confidence = float(
#                 np.max(probabilities) * 100
#             )


#             probability_dict = {}


#             for label, prob in zip(
#                 le.classes_,
#                 probabilities
#             ):

#                 probability_dict[label] = round(
#                     float(prob) * 100,
#                     2
#                 )


#             # ------------------------------------------------
#             # Remove uploaded file
#             # ------------------------------------------------

#             if os.path.exists(file_path):

#                 os.remove(file_path)


#             # ------------------------------------------------
#             # Return Result
#             # ------------------------------------------------

#             return jsonify({

#                 "success": True,

#                 "filename": filename,

#                 "emotion": emotion,

#                 "confidence": round(
#                     confidence,
#                     2
#                 ),

#                 "probabilities":
#                     probability_dict,

#                 "extracted_features":
#                     raw_features,

#                 "sample_rate":
#                     sr,

#                 "duration":
#                     TARGET_DURATION,

#                 "pre_emphasis":
#                     PRE_EMPHASIS,

#                 "waveform_original":
#                     waveform_original,

#                 "waveform_emphasized":
#                     waveform_emphasized,

#                 "waveform_time":
#                     waveform_time

#             }), 200


#         except Exception as e:

#             if os.path.exists(file_path):

#                 os.remove(file_path)


#             return jsonify({
#                 "error": str(e)
#             }), 500


#     return jsonify({
#         "error": "Invalid file format"
#     }), 400
# =========================================================
# Extract Features
# =========================================================
# =========================================================
# Prediction
# =========================================================

# @app.route(
#     "/predict",
#     methods=["POST"]
# )
# def predict():

#     if "feature_vector" not in request.form:

#         return jsonify({
#             "error":
#                 "No extracted feature vector provided. "
#                 "Please extract features first."
#         }), 400

#     try:

#         # ------------------------------------------------
#         # Get Feature Vector
#         # ------------------------------------------------

#         feature_vector_json = request.form.get(
#             "feature_vector"
#         )

#         features = np.array(
#             json.loads(feature_vector_json),
#             dtype=float
#         )

#         # ------------------------------------------------
#         # Check Feature Dimension
#         # ------------------------------------------------

#         if features.shape[0] != 360:

#             return jsonify({
#                 "error":
#                     "Invalid feature vector. "
#                     "Expected 360 features."
#             }), 400

#         # ------------------------------------------------
#         # Reshape
#         # ------------------------------------------------

#         X = features.reshape(
#             1,
#             -1
#         )

#         # ------------------------------------------------
#         # Standardization
#         # ------------------------------------------------

#         X_scaled = scaler.transform(
#             X
#         )

#         # ------------------------------------------------
#         # Prediction
#         # ------------------------------------------------

#         prediction = mlp.predict(
#             X_scaled
#         )

#         emotion = le.inverse_transform(
#             prediction
#         )[0]

#         # ------------------------------------------------
#         # Probability
#         # ------------------------------------------------

#         probabilities = mlp.predict_proba(
#             X_scaled
#         )[0]

#         confidence = float(
#             np.max(probabilities) * 100
#         )

#         probability_dict = {}

#         for label, prob in zip(
#             le.classes_,
#             probabilities
#         ):

#             probability_dict[label] = round(
#                 float(prob) * 100,
#                 2
#             )

#         # ------------------------------------------------
#         # Return Result
#         # ------------------------------------------------

#         return jsonify({

#             "success": True,

#             "emotion": emotion,

#             "confidence": round(
#                 confidence,
#                 2
#             ),

#             "probabilities":
#                 probability_dict

#         }), 200

#     except Exception as e:

#         return jsonify({
#             "error": str(e)
#         }), 500

# =========================================================
# Run Flask
# =========================================================

# if __name__ == "__main__":

#     app.run(
#         host="127.0.0.1",
#         port=5000,
#         debug=True
#     )


