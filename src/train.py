import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def train_model():
    # Set experiment name
    mlflow.set_experiment("Student Score Predictor")

    # Load cleaned data
    df = pd.read_csv("data/processed/cleaned.csv")
    X = df.drop("score", axis=1)
    y = df["score"]

    # Save column names
    feature_columns = list(X.columns)
    print("Trained columns:", feature_columns)
    joblib.dump(feature_columns, "models/columns.pkl")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        # Initialize model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # Log params & metrics
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)

        # Save and log model
        joblib.dump(model, "models/model.pkl")
        mlflow.sklearn.log_model(model, artifact_path="student_model")
        mlflow.log_artifact("models/model.pkl")

        print(f"✅ Model trained. R² Score: {r2:.2f} | MSE: {mse:.2f}")
        print("📦 Model saved to models/model.pkl and tracked via MLflow.")


if __name__ == "__main__":
    train_model()
