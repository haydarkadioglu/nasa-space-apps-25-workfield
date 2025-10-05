
# PRODUCTION PREDICTION FUNCTION
import joblib
import pandas as pd
import numpy as np

def predict_exoplanet(input_features):
    """
    Predict exoplanet probability for new data

    Args:
        input_features: DataFrame or array with same features as training data

    Returns:
        dict: prediction results with probability and classification
    """
    # Load the trained model
    model = joblib.load('best_exoplanet_model_stackingclassifier.joblib')

    # Make prediction
    prediction_proba = model.predict_proba(input_features)[:, 1]
    prediction_class = model.predict(input_features)

    return {
        'exoplanet_probability': prediction_proba[0] if len(prediction_proba) == 1 else prediction_proba,
        'is_exoplanet': bool(prediction_class[0]) if len(prediction_class) == 1 else prediction_class,
        'confidence': 'High' if max(prediction_proba) > 0.8 else 'Medium' if max(prediction_proba) > 0.6 else 'Low'
    }

# Example usage:
# result = predict_exoplanet(new_data)
# print(f"Exoplanet probability: {result['exoplanet_probability']:.3f}")
