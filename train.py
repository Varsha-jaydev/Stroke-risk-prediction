
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocess import load_data, create_preprocessor


DATA_PATH = "data/healthcare-dataset-stroke-data.csv"
MODEL_PATH = "models/stroke_model.pkl"


def evaluate_model(model, X_test, y_test, model_name):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"PR-AUC : {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return pr_auc


def main():
    os.makedirs("models", exist_ok=True)

    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = create_preprocessor()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    best_model = None
    best_score = -1
    best_name = None

    for name, classifier in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        score = evaluate_model(
            pipeline,
            X_test,
            y_test,
            name,
        )

        if score > best_score:
            best_score = score
            best_model = pipeline
            best_name = name

    joblib.dump(best_model, MODEL_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model : {best_name}")
    print(f"PR-AUC     : {best_score:.4f}")
    print(f"Saved to   : {MODEL_PATH}")


if __name__ == "__main__":
    main()