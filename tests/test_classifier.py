# tests/test_classifier.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier import SimpleTextClassifier

def test_prediction():
    classifier = SimpleTextClassifier()

    # Expanded training data with stronger context
    texts = [
        "happy feedback",
        "very good service",
        "excellent experience",
        "great customer support",
        "satisfied client",
        "bad response",
        "no feedback",
        "customer unhappy",
        "terrible support",
        "poor service"
    ]
    labels = [
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative"
    ]

    classifier.train(texts, labels)

    assert classifier.predict(["happy customer"])[0] == "positive"
    assert classifier.predict(["no feedback"])[0] == "negative"
