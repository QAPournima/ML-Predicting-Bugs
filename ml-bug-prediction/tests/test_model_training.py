import unittest
from src.model_training import train_model, save_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model_path = 'test_model.pkl'

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_save_model(self):
        model = train_model(self.X_train, self.y_train)
        save_model(model, self.model_path)
        self.assertTrue(os.path.exists(self.model_path))

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()