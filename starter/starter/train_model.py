# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
df = pd.read_csv("../data/census.csv")
df.columns = df.columns.str.replace(' ', '')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
model = train_model(X_train, y_train)

pickle.dump(model, open('model/model.pkl', 'wb'))