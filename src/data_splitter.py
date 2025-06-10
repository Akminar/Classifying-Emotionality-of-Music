import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/merged_features.csv")

x = df.drop(columns=["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std", "label"])
y = df["label"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42, stratify=y
)

pd.DataFrame(x_train).to_csv("../data/processed_data/train_features.csv", index=False)
pd.DataFrame(x_test).to_csv("../data/processed_data/test_features.csv", index=False)
pd.Series(y_train).to_csv("../data/processed_data/train_labels.csv", index=False)
pd.Series(y_test).to_csv("../data/processed_data/test_labels.csv", index=False)

print("Train/test split complete")