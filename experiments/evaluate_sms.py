from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data(path):
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            label, text = line.split("\t", 1)
            labels.append(label)
            texts.append(text)

    return texts, labels


baseline_texts, baseline_labels = load_data(
    "data/sms_spam_train_500_safe.txt"
)

eda_texts, eda_labels = load_data(
    "data/sms_spam_train_500_augmented.txt"
)

test_texts, test_labels = load_data(
    "data/sms_spam_test_500.txt"
)


def run_experiment(train_texts, train_labels, name):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, train_labels)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(test_labels, predictions)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Training examples: {len(train_texts)}")
    print(f"Test examples: {len(test_texts)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(test_labels, predictions))

    return accuracy


baseline_accuracy = run_experiment(
    baseline_texts,
    baseline_labels,
    "SMS BASELINE - ORIGINAL TRAINING DATA"
)

eda_accuracy = run_experiment(
    eda_texts,
    eda_labels,
    "SMS EDA - AUGMENTED TRAINING DATA"
)

difference = eda_accuracy - baseline_accuracy

print("\n" + "=" * 50)
print("FINAL SMS COMPARISON")
print("=" * 50)

print(f"Baseline accuracy: {baseline_accuracy * 100:.2f}%")
print(f"EDA accuracy:      {eda_accuracy * 100:.2f}%")
print(f"Difference:        {difference * 100:+.2f} percentage points")

if difference > 0:
    print("Result: EDA improved accuracy.")
elif difference < 0:
    print("Result: EDA reduced accuracy.")
else:
    print("Result: Accuracy was unchanged.")
