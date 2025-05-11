from classifier import SimpleTextClassifier

def test_prediction():
    classifier = SimpleTextClassifier()
    texts = ["happy feedback", "bad response"]
    labels = ["positive", "negative"]
    classifier.train(texts, labels)

    assert classifier.predict(["happy customer"])[0] == "positive"
    assert classifier.predict(["no feedback"])[0] == "negative"
