
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

def load_dataset(file_path: str):
    """
    Load dataset from a CSV file.
    
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the dataset.
    """
    return pd.read_csv(file_path)

def separate_features_and_target(df: pd.DataFrame, target_column: str):
    """
    Separate features and target variable from the DataFrame.
    
    :param df: DataFrame containing the dataset.
    :param target_column: Name of the target column.
    :return: Tuple of features DataFrame and target Series.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def train_test_split_dataset(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and testing sets.
    
    :param X: Features DataFrame.
    :param y: Target Series.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.
    :return: Tuple of training and testing sets for features and target.
    """
    # X_train, X_test, y_train, y_test
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a Linear Regression model.
    
    :param X_train: Training features DataFrame.
    :param y_train: Training target Series.
    :return: Trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate the trained model using RMSE and R^2 score.
    
    :param model: Trained Linear Regression model.
    :param X_test: Testing features DataFrame.
    :param y_test: Testing target Series.
    :return: Tuple of RMSE and R^2 score.
    """
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, r2

def main(file_path: str, target_column: str, processed_columns: list = None):
    df = load_dataset(file_path)
    df = df[processed_columns] if processed_columns else df

    print(" Dataset loaded successfully.")
    print(f" Dataset shape: {df.shape}")
    print(f" Columns: {df.columns.tolist()}")

    X, y = separate_features_and_target(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)

    model = train_model(X_train, y_train)

    rmse, r2 = evaluate_model(model, X_test, y_test)

    print(" Model Evaluation:")
    print(f"    Coefficients: {model.coef_}")
    print(f"    Training Score: {model.score(X_train, y_train)}")
    print(f"    RMSE: {rmse}")
    print(f"    R² Score: {r2}")

