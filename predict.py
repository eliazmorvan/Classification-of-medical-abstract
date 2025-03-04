import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_model_and_tokenizer(model_path="final_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4)
    model.eval()  # Put model on eval mode
    return model, tokenizer


def predict(model, tokenizer, test_data, max_length=512):
    encodings = tokenizer.batch_encode_plus(
        test_data["medical_abstract"].tolist(),
        max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits

    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    return predictions


def evaluate_performance(true_labels, predictions):
    print("\nAccuracy:", accuracy_score(true_labels, predictions))
    print("\nClassification report:\n", classification_report(true_labels, predictions))
    print("\nConfusion matrix:\n", confusion_matrix(true_labels, predictions))

# Main
def main(input_file, output_file, model_path="path_to_saved_model"):

    test_data = pd.read_csv(input_file)
    test_data = test_data[test_data["condition_label"] != "general pathological conditions"]

    model, tokenizer = load_model_and_tokenizer(model_path)

    predictions = predict(model, tokenizer, test_data)

    label_mapping = {
        0: "cardiovascular diseases",
        1: "digestive system diseases",
        2: "nervous system diseases",
        3: "neoplasms"
    }
    predictions_labels = [label_mapping[pred] for pred in predictions]

    test_data["predicted_condition_label"] = predictions_labels

    test_data.to_csv(output_file, index=False)
    print(f"\n Predictions saved to {output_file}")

    true_labels = test_data["condition_label"]
    evaluate_performance(true_labels, predictions_labels)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict conditions and evaluate performance")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("output_file", type=str, help="Path to save the output CSV file")
    parser.add_argument("model_path", type=str, help="Path to the saved model")
    args = parser.parse_args()

    main(args.input_file, args.output_file, model_path=args.model_path)