import pandas as pd
import os

ANNOTATION_FILES = [
    "data/annotations/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_1_2000.csv",
    "data/annotations/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_2000_2058.csv"
]

FEATURE_DIR = "data/features"
OUTPUT = "data/merged_features.csv"

dfs = []
for f in ANNOTATION_FILES:
    df = pd.read_csv(f)
    dfs.append(df)
labels_df = pd.concat(dfs, ignore_index=True)

labels_df['song_id'] = labels_df['song_id'].astype(str).str.zfill(4)

def is_focus_friendly(v, a):
    return int((v >= 0) and (-0.5 <= a <= 0.2))

labels_df['label'] = labels_df.apply(lambda row: is_focus_friendly(row['valence_mean'], row['arousal_mean']), axis=1)

feature_dfs = []
skipped = []

for song_id in labels_df['song_id']:
    feature_path = os.path.join(FEATURE_DIR, f"{song_id}.csv")
    try:
        features = pd.read_csv(feature_path)
        features['song_id'] = song_id
        feature_dfs.append(features)
    except Exception as e:
        print(f"Could not read {song_id}.csv: {e}")
        skipped.append(song_id)

features_df = pd.concat(feature_dfs, ignore_index=True)

merged = pd.merge(labels_df, features_df, on='song_id')

merged.to_csv(OUTPUT, index=False)
print(f"Merged dataset saved to {OUTPUT}")