# This file contains unit tests for the models module.

import unittest
from src.models.gnn import GNN
from src.models.temporal import TemporalModel

class TestGNN(unittest.TestCase):
    def setUp(self):
        self.model = GNN()

    def test_forward(self):
        # Add test for forward method
        pass

    def test_train(self):
        # Add test for train method
        pass

class TestTemporalModel(unittest.TestCase):
    def setUp(self):
        self.model = TemporalModel()

    def test_forward(self):
        # Add test for forward method
        pass

    def test_predict(self):
        # Add test for predict method
        pass

if __name__ == '__main__':
    unittest.main()