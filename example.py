from tokenizer import tokenize
from classifier import SimpleTextClassifier

# Sample training data
texts = [
    "survey response yes",
    "no comment survey",
    "feedback is positive",
    "response is negative",
    "happy customer feedback"
]
labels = ["positive", "negative", "positive", "negative", "positive"]

# Initialize and train classifier
classifier = SimpleTextClassifier()
classifier.train(texts, labels)

# Predict new data
new_texts = ["no response", "positive comment", "unhappy survey"]
predictions = classifier.predict(new_texts)

# Display predictions
for text, label in zip(new_texts, predictions):
    print(f"'{text}' => Predicted: {label}")

# Tokenize example
for t in new_texts:
    print(f"Tokens for '{t}': {tokenize(t)}")
