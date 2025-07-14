from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import torch
import numpy as np

app = Flask(__name__)

# Model paths
model_dir = r"E:/grade project/model/new new/dirc/final large"
safetensors_path = f"{model_dir}/model.safetensors"

# Load model and tokenizer
config = AutoConfig.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=False)
model = AutoModelForSequenceClassification.from_config(config)
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict, strict=False)
model.eval()


def analyze_feedbacks(feedbacks):
    total_feedbacks = len(feedbacks)
    results = []
    sentiment_count = {"Negative": 0, "Neutral": 0, "Positive": 0}

    for feedback in feedbacks:
        inputs = tokenizer(feedback, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]

        sentiment_idx = np.argmax(scores)
        sentiment_label = ["Negative", "Neutral", "Positive"][sentiment_idx]
        sentiment_count[sentiment_label] += 1

        results.append({
            "feedback": feedback,
            "sentiment": sentiment_label
        })

    percentages = {
        "Negative": round((sentiment_count["Negative"] / total_feedbacks) * 100, 2),
        "Neutral": round((sentiment_count["Neutral"] / total_feedbacks) * 100, 2),
        "Positive": round((sentiment_count["Positive"] / total_feedbacks) * 100, 2)
    }

    summary = {
        "total_feedbacks": total_feedbacks,
        "most_common_sentiment": max(sentiment_count, key=sentiment_count.get),
        "least_common_sentiment": min(sentiment_count, key=sentiment_count.get)
    }

    return results, percentages, summary


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    feedbacks = data.get("feedback", [])
    event_id = data.get("event_id")

    if not feedbacks or not event_id:
        return jsonify({"error": "Feedback or Event ID missing"}), 400

    results, percentages, summary = analyze_feedbacks(feedbacks)

    response_data = {
        "event_id": event_id,
        "message": "✅ Feedback analyzed successfully.",
        "results": results,
        "percentages": percentages,
        "summary": summary
    }

    return jsonify(response_data), 200


if __name__ == "__main__":
    app.run(host='192.168.191.16', debug=True, port=5001)
