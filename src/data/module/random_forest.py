def main(file_path, target_column, processed_columns):
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # Load the dataset
    df = pd.read_csv(file_path)

    # Extract features and target
    X = df[processed_columns].drop(columns=[target_column])
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"✅ R² Score: {r2:.4f}")
    print(f"✅ RMSE: {rmse:.4f}")

    # Save model
    with open("model.pickle", "wb") as f:
        pickle.dump(model, f)
    print("💾 Model saved as model.pickle")
