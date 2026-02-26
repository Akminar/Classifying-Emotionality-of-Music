import argparse
import os

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
ANNOTATION_FILES = [
    os.path.join(BASE_DIR, "data/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"),
    os.path.join(BASE_DIR, "data/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv"),
]
AUDIO_DIR = os.path.join(BASE_DIR, "data/MEMD_audio")
MERGED_CSV = os.path.join(BASE_DIR, "data/merged_features.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed_data")

METADATA_COLS = ["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std", "label"]


def load_annotations():
    dfs = []
    for f in ANNOTATION_FILES:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    labels_df = pd.concat(dfs, ignore_index=True)
    labels_df = labels_df[["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"]]
    labels_df["song_id"] = labels_df["song_id"].astype(str)
    print(f"Loaded {len(labels_df)} annotations")
    return labels_df


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_means = np.mean(chroma, axis=1)

    features = {
        "tempo": float(np.atleast_1d(tempo)[0]),
        "zcr": zcr,
        "spectral_centroid": spectral_centroid,
    }
    for i in range(13):
        features[f"mfcc_{i+1}"] = mfcc_means[i]
    for i in range(12):
        features[f"chroma_{i+1}"] = chroma_means[i]

    return features


def extract_all_features(labels_df):
    feature_rows = []
    skipped = []

    for song_id in tqdm(labels_df["song_id"], desc="Extracting features"):
        audio_path = os.path.join(AUDIO_DIR, f"{song_id}.mp3")
        try:
            features = extract_features(audio_path)
            features["song_id"] = song_id
            feature_rows.append(features)
        except Exception as e:
            print(f"Could not process {song_id}.mp3: {e}")
            skipped.append(song_id)

    if skipped:
        print(f"Skipped {len(skipped)} songs: {skipped}")

    return pd.DataFrame(feature_rows)


def merge_features(labels_df, features_df):
    merged = pd.merge(labels_df, features_df, on="song_id")

    merged["label"] = merged.apply(
        lambda row: int((row["valence_mean"] >= 5) and (0 <= row["arousal_mean"] <= 5)),
        axis=1,
    )

    merged.to_csv(MERGED_CSV, index=False)
    print(f"Merged dataset saved to {MERGED_CSV}")
    print(f"Shape: {merged.shape}")
    print(f"Label distribution:\n{merged['label'].value_counts()}")
    return merged


def split_data(merged_csv=MERGED_CSV):
    df = pd.read_csv(merged_csv)

    x = df.drop(columns=METADATA_COLS)
    y = df["label"]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42, stratify=y,
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pd.DataFrame(x_train).to_csv(os.path.join(PROCESSED_DIR, "train_features.csv"), index=False)
    pd.DataFrame(x_test).to_csv(os.path.join(PROCESSED_DIR, "test_features.csv"), index=False)
    pd.Series(y_train).to_csv(os.path.join(PROCESSED_DIR, "train_labels.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(PROCESSED_DIR, "test_labels.csv"), index=False)
    print("Train/test split complete")


def run_pipeline():
    labels_df = load_annotations()
    features_df = extract_all_features(labels_df)
    merge_features(labels_df, features_df)
    split_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing pipeline")
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=["all", "extract", "merge", "split"],
        help="Pipeline step to run (default: all)",
    )
    args = parser.parse_args()

    if args.step == "all":
        run_pipeline()
    elif args.step == "extract":
        labels_df = load_annotations()
        features_df = extract_all_features(labels_df)
        merge_features(labels_df, features_df)
    elif args.step == "merge":
        labels_df = load_annotations()
        features_df = extract_all_features(labels_df)
        merge_features(labels_df, features_df)
    elif args.step == "split":
        split_data()
