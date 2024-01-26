import unittest
import numpy as np
import pickle
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

class TestComputeModelMetrics(unittest.TestCase):
    def test_compute_model_metrics(self):
        # Test case 1: Positive values
        actual = compute_model_metrics([1, 1, 1, 1], [1, 1, 1, 1])
        expected = (1.0, 1.0, 1.0)
        self.assertEqual(actual, expected)

        # Test case 2: Negative values
        actual = compute_model_metrics([-1, -1, -1, -1], [-1, -1, -1, -1])
        expected = (1.0, 1.0, 1.0)
        self.assertEqual(actual, expected)

        # Test case 3: Mixed values
        actual = compute_model_metrics([1, 0, 1, 0], [1, 1, 0, 0])
        expected = (0.5, 0.5, 0.5)
        self.assertEqual(actual, expected)

class TestTrainModel(unittest.TestCase):
    def test_train_model(self):
        # Test case 1: Positive values
        X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y_train = np.array([0, 1, 0])
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

        # Test case 2: Negative values
        X_train = np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])
        y_train = np.array([1, 0, 1])
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)

class TestInference(unittest.TestCase):
    def test_inference(self):
        # Load the model from file
        with open('model/model.pkl', 'rb') as file:
            model = pickle.load(file)
        X = np.random.rand(6513, 108)
        y_pred = inference(model, X)
        self.assertIsNotNone(y_pred)

if __name__ == '__main__':

    unittest.main()