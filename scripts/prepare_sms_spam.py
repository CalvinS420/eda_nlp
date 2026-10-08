import os
import io
import zipfile
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

print("Downloading SMS Spam Collection...")

response = requests.get(URL, timeout=60)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    print("Files in archive:", z.namelist())

    with z.open("SMSSpamCollection") as f:
        df = pd.read_csv(
            f,
            sep="\t",
            names=["label", "text"],
            encoding="utf-8"
        )

print("\nFull dataset:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

# Reproducible 500 train / 500 test split
train_df, test_df = train_test_split(
    df,
    train_size=500,
    test_size=500,
    random_state=42,
    stratify=df["label"]
)

os.makedirs("data", exist_ok=True)

train_df.to_csv(
    "data/sms_spam_train_500.csv",
    index=False
)

test_df.to_csv(
    "data/sms_spam_test_500.csv",
    index=False
)

# Format for the original EDA augment.py:
# label<TAB>sentence

with open(
    "data/sms_spam_train_500.txt",
    "w",
    encoding="utf-8"
) as f:

    for _, row in train_df.iterrows():

        text = str(row["text"]).replace("\n", " ").replace("\t", " ")

        f.write(
            f"{row['label']}\t{text}\n"
        )

with open(
    "data/sms_spam_test_500.txt",
    "w",
    encoding="utf-8"
) as f:

    for _, row in test_df.iterrows():

        text = str(row["text"]).replace("\n", " ").replace("\t", " ")

        f.write(
            f"{row['label']}\t{text}\n"
        )

print("\nPrepared:")
print("Training:", len(train_df))
print("Testing:", len(test_df))

print("\nTraining labels:")
print(train_df["label"].value_counts())

print("\nTesting labels:")
print(test_df["label"].value_counts())

print("\nSMS Spam preparation complete.")
