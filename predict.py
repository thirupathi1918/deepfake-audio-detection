import sys
import numpy as np
import librosa
import joblib
import os

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=5.0)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        return np.hstack([mfccs, mel, chroma]).reshape(1, -1)
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    
    if not os.path.exists('deepfake_detector_model.pkl'):
        print("❌ Model artifact file 'deepfake_detector_model.pkl' not found in this directory.")
        print("Please place the downloaded file from Kaggle here before testing.")
        sys.exit(1)
        
    try:
        payload = joblib.load('deepfake_detector_model.pkl')
        model = payload['model']
        threshold = payload['eer_threshold']
    except Exception as e:
        print(f"❌ Error loading the model payload: {e}")
        sys.exit(1)
        
    features = extract_audio_features(audio_path)
    if features is not None:
        prob = model.predict_proba(features)[0, 1]
        label = "Deepfake (AI-Generated)" if prob >= threshold else "Genuine (Human)"
        confidence = prob if prob >= threshold else (1 - prob)
        
        print(f"\n🎧 File: {audio_path}")
        print(f"Result: {label}")
        print(f"Confidence Score: {confidence*100:.2f}%")

if __name__ == "__main__":
    main()