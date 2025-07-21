from flask import Flask, request, jsonify
from preprocess import clean_text
from model import FakeNewsModel

app = Flask(__name__)
model = FakeNewsModel()

# Sample training data
texts = ["This is real news", "This is fake"]
labels = ["real", "fake"]
cleaned = [clean_text(t) for t in texts]
model.train(cleaned, labels)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = clean_text(data['text'])
    result = model.predict(text)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)