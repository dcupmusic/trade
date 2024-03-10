import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Read data from CSV file
def read_data(file_path):
    return pd.read_csv(file_path)

# Step 2: Preprocess data and train a machine learning model
def train_model(data):
    # Assuming 'signal' is the target variable and the rest are features
    X = data[['Timestamp', 'Price', 'RSI']]
    y = data['Signal']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree classifier (you can replace this with any other model)
    model = DecisionTreeClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

    return model

# Step 3: Save the trained model
def save_model(model, model_filename='trained_model.joblib'):
    joblib.dump(model, model_filename)
    print(f'Model saved as {model_filename}')

# Main script
if __name__ == "__main__":
    # Replace 'your_data.csv' with the actual file path
    file_path = 'Performance-test.csv'

    # Step 1: Read data
    data = read_data(file_path)

    # Step 2: Train model
    trained_model = train_model(data)

    # Step 3: Save model
    save_model(trained_model)
