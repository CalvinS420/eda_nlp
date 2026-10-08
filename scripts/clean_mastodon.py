import pandas as pd
import re
from bs4 import BeautifulSoup

INPUT = "data/mastodon/mastodon_raw.csv"
OUTPUT = "data/mastodon/mastodon_clean.csv"

df = pd.read_csv(INPUT)

def clean_text(html, source_hashtag):
    if pd.isna(html):
        return ""

    # Remove HTML tags
    text = BeautifulSoup(html, "html.parser").get_text(" ")

    # Remove URLs
    text = re.sub(r'https?://\S+', ' ', text)

    # Remove @mentions
    text = re.sub(r'@\w+(?:@\S+)?', ' ', text)

    # Remove hashtags entirely to avoid leaking the label
    text = re.sub(r'#\w+', ' ', text)

    # Remove source hashtag as an extra safeguard
    text = re.sub(
        rf'\b{re.escape(str(source_hashtag))}\b',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # Keep letters, digits, spaces and basic punctuation
    text = re.sub(r'[^A-Za-z0-9\s.,!?\'"-]', ' ', text)

    # Collapse repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


df["clean_text"] = df.apply(
    lambda row: clean_text(
        row["content_html"],
        row["source_hashtag"]
    ),
    axis=1
)

# Remove empty / extremely short posts
df = df[df["clean_text"].str.len() >= 20].copy()

# Remove duplicate cleaned posts
df = df.drop_duplicates(subset="clean_text")

# Add annotation columns for next stage
df["manual_label"] = ""
df["annotation_status"] = "unreviewed"

df.to_csv(OUTPUT, index=False)

print("=" * 50)
print("MASTODON CLEANING COMPLETE")
print("=" * 50)
print("Raw rows: 179")
print("Clean rows:", len(df))
print("\nRows by provisional label:")
print(df["provisional_label"].value_counts())
print(f"\nSaved to: {OUTPUT}")

print("\nExample cleaned posts:\n")
for i, row in df.head(5).iterrows():
    print(f"[{row['provisional_label']}]")
    print(row["clean_text"])
    print()
