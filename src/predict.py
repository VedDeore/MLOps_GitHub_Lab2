import joblib
import numpy as np

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        numpy.ndarray: Predicted class labels.
    """
    model = joblib.load("../model/digits_model.pkl")
    X = np.array(X)
    return model.predict(X)
