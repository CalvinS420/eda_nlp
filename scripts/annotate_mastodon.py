import pandas as pd

FILE = "data/mastodon/mastodon_annotation_30.csv"

df = pd.read_csv(FILE)

# Make sure empty cells are strings
df["manual_label"] = df["manual_label"].fillna("")
df["annotation_status"] = df["annotation_status"].fillna("unreviewed")
df["notes"] = df["notes"].fillna("")

label_map = {
    "1": "technology",
    "2": "photography",
    "3": "music",
    "4": "unclear"
}

print("\nMastodon Manual Annotation")
print("==========================")
print("1 = technology")
print("2 = photography")
print("3 = music")
print("4 = unclear")
print("s = skip")
print("q = quit")
print()

for i, row in df.iterrows():

    if row["annotation_status"] == "reviewed":
        continue

    print("=" * 80)
    print(f"Row {i + 1} / {len(df)}")
    print(f"Provisional label: {row['provisional_label']}")
    print()
    print(row["clean_text"])
    print()

    choice = input("Your label [1/2/3/4/s/q]: ").strip().lower()

    if choice == "q":
        print("\nStopping. Progress has been saved.")
        break

    if choice == "s":
        continue

    if choice not in label_map:
        print("Invalid choice. Skipping this row.")
        continue

    df.at[i, "manual_label"] = label_map[choice]
    df.at[i, "annotation_status"] = "reviewed"

    # Save after EVERY annotation so progress is never lost
    df.to_csv(FILE, index=False)

    print(f"Saved as: {label_map[choice]}")

# Save once more before exiting
df.to_csv(FILE, index=False)

reviewed = (df["annotation_status"] == "reviewed").sum()

print("\n==========================")
print("ANNOTATION PROGRESS")
print("==========================")
print(f"Reviewed: {reviewed}/{len(df)}")

if reviewed > 0:
    print("\nManual labels:")
    print(
        df[df["annotation_status"] == "reviewed"]
        ["manual_label"]
        .value_counts()
    )
