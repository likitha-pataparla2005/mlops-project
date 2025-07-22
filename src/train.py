import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib


def train_model():
    mlflow.set_experiment("Student Score Predictor")

    df = pd.read_csv("data/processed/cleaned.csv")
    X = df.drop("score", axis=1)
    print("Trained columns:", list(X.columns))
    joblib.dump(list(X.columns), "models/columns.pkl")
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("r2_score", r2)
        joblib.dump(model, "models/model.pkl")
        mlflow.log_artifact("models/model.pkl")

        print(f"✅ Model trained. R² Score: {r2:.2f}")
        print("📦 Model saved to models/model.pkl")


if __name__ == "__main__":
    train_model()
