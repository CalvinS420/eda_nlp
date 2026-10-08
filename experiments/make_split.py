from sklearn.model_selection import train_test_split

input_file = "data/sst2_train_500.txt"

labels = []
texts = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        label, text = line.split("\t", 1)
        labels.append(label)
        texts.append(text)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

with open("data/sst2_train_400.txt", "w", encoding="utf-8") as f:
    for label, text in zip(train_labels, train_texts):
        f.write(f"{label}\t{text}\n")

with open("data/sst2_test_100.txt", "w", encoding="utf-8") as f:
    for label, text in zip(test_labels, test_texts):
        f.write(f"{label}\t{text}\n")

print("Training examples:", len(train_texts))
print("Test examples:", len(test_texts))
