import argparse
import numpy as np
from joblib import load

MIN_CONFIDENCE = 0.7
K = 2

# Original LABELS with + levels
LABELS_WITH_PLUS = {
    0.0: "A1",
    0.5: "A1+",
    1.0: "A2",
    1.5: "A2+",
    2.0: "B1",
    2.5: "B1+",
    3.0: "B2",
    3.5: "B2+",
    4.0: "C1",
    4.5: "C1+",
    5.0: "C2",
    5.5: "C2+",
}

# Standard 6-level CEFR system
LABELS = {
    0.0: "A1",
    1.0: "A2",
    2.0: "B1",
    3.0: "B2",
    4.0: "C1",
    5.0: "C2",
}

def round_to_standard_level(value):
    """Round a numeric value to the nearest standard CEFR level."""
    # Ensure the value is within bounds
    value = max(0.0, min(5.0, value))
    # Round to nearest whole number
    rounded = round(value)
    return float(rounded)

# Function to convert plus levels to standard levels
def convert_to_standard_level(level):
    if "+" in level:
        base_level = level[:-1]
        next_level_map = {
            "A1": "A2",
            "A2": "B1",
            "B1": "B2",
            "B2": "C1",
            "C1": "C2",
            "C2": "C2"  # C2+ remains C2
        }
        # If confidence in + level is high, round up to next level
        if np.random.random() > 0.5:  # You can adjust this threshold
            return next_level_map[base_level]
        return base_level
    return level

class Model:
    def __init__(self, model_path, use_plus_levels=False):
        self.model = load(model_path)
        self.use_plus_levels = use_plus_levels

    def predict(self, data):
        probas = self.model.predict_proba(data)
        preds = [self._get_pred(p) for p in probas]
        probas = [self._label_probabilities(p) for p in probas]
        return preds, probas

    def predict_decode(self, data):
        preds, probas = self.predict(data)
        if self.use_plus_levels:
            preds = [LABELS_WITH_PLUS[p] for p in preds]
        else:
            # Round to nearest standard level first
            preds = [round_to_standard_level(p) for p in preds]
            # Then convert to CEFR label
            preds = [LABELS[p] for p in preds]
        return preds, probas

    def _get_pred(self, probabilities):
        if probabilities.max() < MIN_CONFIDENCE:
            return np.mean(probabilities.argsort()[-K:])
        else:
            return probabilities.argmax()

    def decode_label(self, encoded_label):
        if self.use_plus_levels:
            return LABELS_WITH_PLUS[encoded_label]
        rounded = round_to_standard_level(encoded_label)
        return LABELS[rounded]

    def _label_probabilities(self, probas):
        labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        return {label: float(proba) for label, proba in zip(labels, probas)}


def parse_text_files():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text_files", nargs="+", default=[])
    args = parser.parse_args()

    texts = []
    for path in args.text_files:
        with open(path, "r") as f:
            texts.append(f.read())
    return texts


if __name__ == "__main__":
    texts = parse_text_files()
    if len(texts) == 0:
        raise Exception("Specify one or more documents to evaluate.")

    model = Model("cefr_predictor/models/xgboost.joblib")
    preds, probas = model.predict_decode(texts)

    results = []
    for text, pred, proba in zip(texts, preds, probas):
        row = {"text": text, "level": pred, "scores": proba}
        results.append(row)

    print(results)
