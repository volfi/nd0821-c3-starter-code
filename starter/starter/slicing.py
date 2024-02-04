import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from ml.model import train_model, compute_model_metrics
from ml.data import process_data


def slice_data(df, feature):
    """ Function for calculating descriptive stats on slices of the Iris dataset."""
    # Add code to load in the data.
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
    X_train, y_train, encoder, lb1 = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )


    # Train model
    model = train_model(X_train, y_train)

    for cls in test[feature].unique():
        test_tmp = test[test[feature] == cls]
        X_test, y_test, encoder, lb = process_data(
            test_tmp, categorical_features=cat_features, label="salary", training=False, encoder=encoder)
        
        y_test_bin = lb1.fit_transform(y_test.values).ravel()

        y_pred = model.predict(X_test)
        precision, recall, fbeta = compute_model_metrics(y_test_bin, y_pred)
        print(f"Feature: {feature}, Class {cls}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Fbeta: {fbeta:.4f}")


if __name__ == '__main__':
       df = pd.read_csv("../data/census.csv")
       slice_data(df, "education")