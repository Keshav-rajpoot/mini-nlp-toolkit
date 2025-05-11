# classifier.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class SimpleTextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()

    def train(self, texts, labels):
        vectors = self.vectorizer.fit_transform(texts)
        self.classifier.fit(vectors, labels)

    def predict(self, texts):
        vectors = self.vectorizer.transform(texts)
        return self.classifier.predict(vectors)
