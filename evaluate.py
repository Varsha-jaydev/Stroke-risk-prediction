
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

from preprocess import load_data


DATA_PATH = "data/healthcare-dataset-stroke-data.csv"
MODEL_PATH = "models/stroke_model.pkl"


def main():
    X, y = load_data(DATA_PATH)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report")
    print("=" * 60)
    print(classification_report(y_test, predictions, zero_division=0))

    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(
        f"PR-AUC : "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )

    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Stroke", "Stroke"],
    )

    display.plot(cmap="Blues")

    plt.title("Stroke Prediction Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()
