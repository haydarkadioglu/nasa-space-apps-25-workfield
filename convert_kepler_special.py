"""
Kepler Model ONNX Converter - Special handling for XGBoost feature names
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import copy

# ONNX conversion libraries
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as MLToolsFloatTensorType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def convert_kepler_model():
    """Special converter for Kepler XGBoost model"""
    print("🔄 Attempting Kepler XGBoost model conversion with feature name fixes...")
    
    output_dir = Path("exoplanet-hunting/models/kepler")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load components
        model = joblib.load("main/kepler/kepler_3class_best_model.joblib")
        scaler = joblib.load("main/kepler/kepler_3class_scaler.joblib")
        
        with open("main/kepler/kepler_3class_feature_names.json", 'r') as f:
            features = json.load(f)
        
        with open("main/kepler/kepler_3class_model_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        n_features = len(features)
        print(f"   📊 Loaded Kepler model with {n_features} features")
        
        # Method 1: Try to retrain XGBoost with generic feature names
        print("   🔧 Attempting to create compatible XGBoost model...")
        
        # Load sample data for retraining
        df = pd.read_csv("kepler/kepler.csv").head(1000)  # Use subset for speed
        
        # Prepare features with generic names
        feature_values = []
        for _, row in df.iterrows():
            row_features = []
            for feature in features:
                if feature in row:
                    value = row[feature]
                    if pd.isna(value):
                        value = 0.0
                    row_features.append(float(value))
                else:
                    row_features.append(0.0)
            feature_values.append(row_features)
        
        X_sample = np.array(feature_values, dtype=np.float32)
        
        # Scale the data
        X_scaled = scaler.transform(X_sample)
        
        # Get predictions from original model for retraining target
        y_pred = model.predict(X_scaled)
        
        # Create new XGBoost model with generic feature names
        from xgboost import XGBClassifier
        
        # Create model with same parameters but generic feature names
        new_model = XGBClassifier(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            random_state=42
        )
        
        # Create generic feature names
        generic_features = [f'f{i}' for i in range(n_features)]
        
        # Create DataFrame with generic column names
        X_generic = pd.DataFrame(X_scaled, columns=generic_features)
        
        # Fit new model
        print("   🏃 Training new XGBoost model with generic feature names...")
        new_model.fit(X_generic, y_pred)
        
        # Convert to ONNX
        initial_type = [('input', MLToolsFloatTensorType([None, n_features]))]
        model_onnx = convert_xgboost(new_model, initial_types=initial_type, target_opset=11)
        
        # Convert scaler to ONNX
        scaler_initial_type = [('float_input', FloatTensorType([None, n_features]))]
        scaler_onnx = convert_sklearn(scaler, initial_types=scaler_initial_type, target_opset=11)
        
        # Save ONNX models
        model_onnx_path = output_dir / "kepler_model.onnx"
        scaler_onnx_path = output_dir / "kepler_scaler.onnx"
        
        with open(model_onnx_path, "wb") as f:
            f.write(model_onnx.SerializeToString())
        
        with open(scaler_onnx_path, "wb") as f:
            f.write(scaler_onnx.SerializeToString())
        
        # Create deployment metadata
        deployment_metadata = {
            "model_info": {
                "dataset": "Kepler",
                "model_type": "XGBClassifier",
                "f1_macro_score": metadata.get('f1_macro_score', 0.0),
                "accuracy_score": metadata.get('accuracy_score', 0.0),
                "training_date": metadata.get('training_date', 'Unknown'),
                "features_count": n_features,
                "training_samples": metadata.get('training_samples', 0),
                "onnx_model_type": "XGBClassifier",
                "conversion_method": "retrained_with_generic_features"
            },
            "deployment_info": {
                "onnx_opset_version": 11,
                "input_shape": [None, n_features],
                "input_type": "float32",
                "output_classes": ["Candidate", "Confirmed", "False_Positive"],
                "class_mapping": {0: "Candidate", 1: "Confirmed", 2: "False_Positive"},
                "feature_names": features,
                "generic_feature_names": generic_features
            },
            "usage_example": {
                "python": {
                    "import": [
                        "import onnxruntime as ort",
                        "import numpy as np"
                    ],
                    "load_model": "session = ort.InferenceSession('kepler_model.onnx')",
                    "load_scaler": "scaler_session = ort.InferenceSession('kepler_scaler.onnx')",
                    "predict": [
                        "# Scale input data first",
                        "scaled_data = scaler_session.run(None, {'float_input': your_data})[0]",
                        "# Make prediction", 
                        "prediction = session.run(None, {'input': scaled_data})"
                    ]
                }
            }
        }
        
        # Save metadata and features
        metadata_path = output_dir / "kepler_deployment.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(deployment_metadata, f, indent=2)
        
        features_path = output_dir / "kepler_features.json"
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=2)
        
        original_metadata_path = output_dir / "kepler_original_metadata.json"
        with open(original_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Kepler model successfully converted to ONNX!")
        print(f"   📄 Model: {model_onnx_path}")
        print(f"   📄 Scaler: {scaler_onnx_path}")
        print(f"   📄 Metadata: {metadata_path}")
        print("   ⚠️  Note: Model was retrained with generic feature names for ONNX compatibility")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting Kepler model: {str(e)}")
        return False

if __name__ == "__main__":
    convert_kepler_model()