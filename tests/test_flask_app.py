import unittest
from flask import Flask
from flask_app.app import app

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Sentiment Analysis', response.data)

    def test_predict_page(self):
        response = self.app.post('/predict', data={'review': 'This is a great product!'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Positive', response.data)
        
        
if __name__ == '__main__':
    unittest.main()