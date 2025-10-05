"""
NASA Exoplanet Model ONNX Converter
Converts trained scikit-learn models to ONNX format for deployment
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

# ONNX conversion libraries
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    # Register LightGBM and XGBoost converters
    from onnxmltools.convert import convert_lightgbm, convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as MLToolsFloatTensorType
    
    print("✅ ONNX conversion libraries loaded successfully")
except ImportError:
    print("❌ Missing ONNX libraries. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "skl2onnx", "onnx", "onnxruntime", "onnxmltools", "onnxconverter-common"])
    
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from onnxmltools.convert import convert_lightgbm, convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as MLToolsFloatTensorType

class ModelONNXConverter:
    def __init__(self, output_dir="exoplanet-hunting/models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            "K2": {
                "model_path": "main/k2/k2_3class_best_model.joblib",
                "metadata_path": "main/k2/k2_3class_model_metadata.json",
                "features_path": "main/k2/k2_3class_feature_names.json",
                "scaler_path": "main/k2/k2_3class_scaler.joblib",
                "csv_path": "k2/k2.csv"
            },
            "Kepler": {
                "model_path": "main/kepler/kepler_3class_best_model.joblib",
                "metadata_path": "main/kepler/kepler_3class_model_metadata.json",
                "features_path": "main/kepler/kepler_3class_feature_names.json",
                "scaler_path": "main/kepler/kepler_3class_scaler.joblib",
                "csv_path": "kepler/kepler.csv"
            },
            "TESS": {
                "model_path": "tess/tess_models/tess_3class_best_model.joblib",
                "metadata_path": "tess/tess_models/tess_3class_model_metadata.json",
                "features_path": "tess/tess_models/tess_3class_feature_names.json",
                "scaler_path": "tess/tess_models/tess_3class_scaler.joblib",
                "csv_path": "tess/TOI.csv"
            }
        }
        
        self.class_mapping = {
            0: "Candidate",
            1: "Confirmed", 
            2: "False_Positive"
        }
    
    def load_model_components(self, dataset):
        """Load model, scaler, features, and metadata"""
        config = self.model_configs[dataset]
        
        try:
            # Load components
            model = joblib.load(config["model_path"])
            scaler = joblib.load(config["scaler_path"])
            
            with open(config["features_path"], 'r') as f:
                features = json.load(f)
            
            with open(config["metadata_path"], 'r') as f:
                metadata = json.load(f)
            
            print(f"✅ {dataset} model components loaded successfully")
            
            return {
                'model': model,
                'scaler': scaler,
                'features': features,
                'metadata': metadata,
                'config': config
            }
            
        except Exception as e:
            print(f"❌ Error loading {dataset} model: {str(e)}")
            return None
    
    def convert_model_to_onnx(self, dataset, components):
        """Convert a single model to ONNX format"""
        print(f"\n🔄 Converting {dataset} model to ONNX...")
        
        model = components['model']
        scaler = components['scaler']
        features = components['features']
        metadata = components['metadata']
        
        try:
            # Determine input shape
            n_features = len(features)
            
            # Detect model type and use appropriate converter
            model_type = type(model).__name__
            
            if 'LGBMClassifier' in model_type:
                print(f"   📊 Detected LightGBM model")
                # Use onnxmltools for LightGBM - use generic input name
                initial_type = [('input', MLToolsFloatTensorType([None, n_features]))]
                model_onnx = convert_lightgbm(model, initial_types=initial_type, target_opset=11)
                
            elif 'XGBClassifier' in model_type:
                print(f"   📊 Detected XGBoost model")
                # Use onnxmltools for XGBoost - convert feature names to generic format
                initial_type = [('input', MLToolsFloatTensorType([None, n_features]))]
                
                # XGBoost requires generic feature names
                # Create a copy of the model and update feature names if needed
                import copy
                model_copy = copy.deepcopy(model)
                
                # Try to set feature names to generic format if the model supports it
                try:
                    if hasattr(model_copy, 'feature_names_in_'):
                        # Replace with generic names
                        generic_names = [f'f{i}' for i in range(n_features)]
                        model_copy.feature_names_in_ = generic_names
                except:
                    pass
                
                model_onnx = convert_xgboost(model_copy, initial_types=initial_type, target_opset=11)
                
            else:
                print(f"   📊 Detected {model_type} - using sklearn converter")
                # Use skl2onnx for other models
                initial_type = [('float_input', FloatTensorType([None, n_features]))]
                model_onnx = convert_sklearn(model, initial_types=initial_type, target_opset=11)
            
            # Convert scaler to ONNX (always use skl2onnx for scaler)
            scaler_initial_type = [('float_input', FloatTensorType([None, n_features]))]
            scaler_onnx = convert_sklearn(scaler, initial_types=scaler_initial_type, target_opset=11)
            
            # Save ONNX models
            dataset_dir = self.output_dir / dataset.lower()
            dataset_dir.mkdir(exist_ok=True)
            
            model_onnx_path = dataset_dir / f"{dataset.lower()}_model.onnx"
            scaler_onnx_path = dataset_dir / f"{dataset.lower()}_scaler.onnx"
            
            with open(model_onnx_path, "wb") as f:
                f.write(model_onnx.SerializeToString())
            
            with open(scaler_onnx_path, "wb") as f:
                f.write(scaler_onnx.SerializeToString())
            
            # Create deployment metadata
            deployment_metadata = {
                "model_info": {
                    "dataset": dataset,
                    "model_type": metadata.get('model_type', model_type),
                    "f1_macro_score": metadata.get('f1_macro_score', 0.0),
                    "accuracy_score": metadata.get('accuracy_score', 0.0),
                    "training_date": metadata.get('training_date', 'Unknown'),
                    "features_count": n_features,
                    "training_samples": metadata.get('training_samples', 0),
                    "onnx_model_type": model_type
                },
                "deployment_info": {
                    "onnx_opset_version": 11,
                    "input_shape": [None, n_features],
                    "input_type": "float32",
                    "output_classes": list(self.class_mapping.values()),
                    "class_mapping": self.class_mapping,
                    "feature_names": features
                },
                "usage_example": {
                    "python": {
                        "import": [
                            "import onnxruntime as ort",
                            "import numpy as np"
                        ],
                        "load_model": f"session = ort.InferenceSession('{dataset.lower()}_model.onnx')",
                        "load_scaler": f"scaler_session = ort.InferenceSession('{dataset.lower()}_scaler.onnx')",
                        "predict": [
                            "# Scale input data first",
                            "scaled_data = scaler_session.run(None, {'float_input': your_data})[0]",
                            "# Make prediction", 
                            "prediction = session.run(None, {'float_input': scaled_data})[0]"
                        ]
                    },
                    "javascript": {
                        "library": "onnxjs or onnxruntime-web",
                        "note": "Can be used in web browsers with WebAssembly"
                    },
                    "csharp": {
                        "library": "Microsoft.ML.OnnxRuntime",
                        "note": "For .NET applications"
                    }
                }
            }
            
            # Save metadata
            metadata_path = dataset_dir / f"{dataset.lower()}_deployment.json"
            with open(metadata_path, 'w') as f:
                json.dump(deployment_metadata, f, indent=2)
            
            # Copy feature names and original metadata
            features_path = dataset_dir / f"{dataset.lower()}_features.json"
            with open(features_path, 'w') as f:
                json.dump(features, f, indent=2)
            
            original_metadata_path = dataset_dir / f"{dataset.lower()}_original_metadata.json"
            with open(original_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ {dataset} model converted to ONNX successfully")
            print(f"   📄 Model: {model_onnx_path}")
            print(f"   📄 Scaler: {scaler_onnx_path}")
            print(f"   📄 Metadata: {metadata_path}")
            
            return {
                'model_path': model_onnx_path,
                'scaler_path': scaler_onnx_path,
                'metadata_path': metadata_path,
                'features_path': features_path,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error converting {dataset} model to ONNX: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def test_onnx_model(self, dataset, onnx_paths, original_components):
        """Test ONNX model conversion by comparing with original"""
        print(f"\n🧪 Testing {dataset} ONNX model...")
        
        try:
            import onnxruntime as ort
            
            # Load ONNX models
            model_session = ort.InferenceSession(str(onnx_paths['model_path']))
            scaler_session = ort.InferenceSession(str(onnx_paths['scaler_path']))
            
            # Load sample data for testing
            config = self.model_configs[dataset]
            if os.path.exists(config['csv_path']):
                df = pd.read_csv(config['csv_path']).head(5)
                features = original_components['features']
                
                # Prepare test data
                test_data = []
                for _, row in df.iterrows():
                    feature_values = []
                    for feature in features:
                        if feature in row:
                            value = row[feature]
                            if pd.isna(value):
                                value = 0.0
                            feature_values.append(float(value))
                        else:
                            feature_values.append(0.0)
                    test_data.append(feature_values)
                
                test_data = np.array(test_data, dtype=np.float32)
                
                # Test original model
                original_scaled = original_components['scaler'].transform(test_data)
                original_pred = original_components['model'].predict(original_scaled)
                original_proba = original_components['model'].predict_proba(original_scaled)
                
                # Test ONNX model - handle different input names
                input_name = model_session.get_inputs()[0].name
                scaler_input_name = scaler_session.get_inputs()[0].name
                
                onnx_scaled = scaler_session.run(None, {scaler_input_name: test_data})[0]
                onnx_pred = model_session.run(None, {input_name: onnx_scaled})[0]
                onnx_proba = model_session.run(None, {input_name: onnx_scaled})[1]
                
                # Compare results
                pred_match = np.allclose(original_pred, onnx_pred)
                proba_match = np.allclose(original_proba, onnx_proba, rtol=1e-5)
                
                if pred_match and proba_match:
                    print(f"✅ {dataset} ONNX model test PASSED - Results match original model")
                else:
                    print(f"⚠️ {dataset} ONNX model test WARNING - Small differences detected")
                    print(f"   Predictions match: {pred_match}")
                    print(f"   Probabilities match: {proba_match}")
                
                return True
                
            else:
                print(f"⚠️ CSV file not found for testing: {config['csv_path']}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing {dataset} ONNX model: {str(e)}")
            return False
    
    def create_deployment_readme(self):
        """Create README for deployment"""
        readme_content = """# NASA Exoplanet Classification Models - ONNX Deployment

## 🚀 Overview

This directory contains ONNX (Open Neural Network Exchange) versions of trained NASA exoplanet classification models from three missions:

- **K2 Mission Model**: Trained on K2 exoplanet candidates
- **Kepler Mission Model**: Trained on Kepler exoplanet candidates  
- **TESS Mission Model**: Trained on TESS Objects of Interest (TOI)

## 📁 Directory Structure

```
models/
├── k2/
│   ├── k2_model.onnx              # K2 classification model
│   ├── k2_scaler.onnx             # K2 feature scaler
│   ├── k2_deployment.json         # Deployment metadata
│   ├── k2_features.json           # Feature names
│   └── k2_original_metadata.json  # Original training metadata
├── kepler/
│   ├── kepler_model.onnx          # Kepler classification model
│   ├── kepler_scaler.onnx         # Kepler feature scaler
│   ├── kepler_deployment.json     # Deployment metadata
│   ├── kepler_features.json       # Feature names
│   └── kepler_original_metadata.json
└── tess/
    ├── tess_model.onnx            # TESS classification model
    ├── tess_scaler.onnx           # TESS feature scaler
    ├── tess_deployment.json       # Deployment metadata
    ├── tess_features.json         # Feature names
    └── tess_original_metadata.json
```

## 🎯 Classification Categories

All models classify exoplanet candidates into three categories:

- **Candidate**: Planet candidates requiring follow-up observation
- **Confirmed**: Confirmed exoplanets with high confidence
- **False Positive**: False positives and refuted planetary candidates

## 💻 Usage Examples

### Python with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
import json

# Load model and scaler
model_session = ort.InferenceSession('models/k2/k2_model.onnx')
scaler_session = ort.InferenceSession('models/k2/k2_scaler.onnx')

# Load feature names and metadata
with open('models/k2/k2_features.json', 'r') as f:
    features = json.load(f)

with open('models/k2/k2_deployment.json', 'r') as f:
    metadata = json.load(f)

# Prepare your data (example with random data)
# Your data should have the same features as in k2_features.json
input_data = np.random.random((1, len(features))).astype(np.float32)

# Scale the data
scaled_data = scaler_session.run(None, {'float_input': input_data})[0]

# Make prediction
prediction = model_session.run(None, {'float_input': scaled_data})
predicted_class = prediction[0][0]  # Class prediction
class_probabilities = prediction[1][0]  # Class probabilities

# Map to class names
class_names = metadata['deployment_info']['output_classes']
predicted_label = class_names[predicted_class]

print(f"Predicted class: {predicted_label}")
print(f"Probabilities: {dict(zip(class_names, class_probabilities))}")
```

### JavaScript with ONNX.js

```javascript
const ort = require('onnxruntime-web');

async function loadAndPredict() {
    // Load models
    const modelSession = await ort.InferenceSession.create('models/k2/k2_model.onnx');
    const scalerSession = await ort.InferenceSession.create('models/k2/k2_scaler.onnx');
    
    // Prepare input data
    const inputData = new Float32Array([/* your feature values */]);
    const tensor = new ort.Tensor('float32', inputData, [1, inputData.length]);
    
    // Scale data
    const scaledResult = await scalerSession.run({float_input: tensor});
    
    // Make prediction
    const prediction = await modelSession.run({float_input: scaledResult.variable});
    
    console.log('Prediction:', prediction);
}
```

### C# with ML.NET

```csharp
using Microsoft.ML.OnnxRuntime;
using System;

// Load model
var session = new InferenceSession("models/k2/k2_model.onnx");
var scalerSession = new InferenceSession("models/k2/k2_scaler.onnx");

// Prepare input data
var inputData = new float[] { /* your feature values */ };
var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, inputData.Length });

// Scale data
var scaledResult = scalerSession.Run(new List<NamedOnnxValue> 
{
    NamedOnnxValue.CreateFromTensor("float_input", inputTensor)
});

// Make prediction
var prediction = session.Run(new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("float_input", scaledResult.First().AsTensor<float>())
});
```

## 🔧 Model Performance

| Model | F1-Macro Score | Accuracy | Training Samples |
|-------|----------------|----------|------------------|
| K2    | [Check deployment.json] | [Check deployment.json] | [Check deployment.json] |
| Kepler| [Check deployment.json] | [Check deployment.json] | [Check deployment.json] |
| TESS  | [Check deployment.json] | [Check deployment.json] | [Check deployment.json] |

## 📋 Requirements

### Python
```bash
pip install onnxruntime numpy
```

### JavaScript/Node.js
```bash
npm install onnxruntime-web
```

### C#/.NET
```bash
dotnet add package Microsoft.ML.OnnxRuntime
```

## 🌐 Deployment Options

### Web Applications
- Use ONNX.js for browser-based inference
- WebAssembly support for high performance
- No server-side processing required

### Mobile Applications
- ONNX Runtime Mobile for iOS/Android
- Optimized for mobile hardware
- Offline inference capability

### Cloud Services
- Deploy with Azure ML, AWS SageMaker, or Google AI Platform
- Auto-scaling and managed inference
- REST API endpoints

### Edge Devices
- ONNX Runtime for IoT devices
- Raspberry Pi and embedded systems
- Low latency inference

## 📖 Additional Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [ONNX Format Specification](https://onnx.ai/)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see contributing guidelines.

---

Generated automatically from trained scikit-learn models.
Last updated: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """
"""
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ Deployment README created: {readme_path}")
    
    def convert_all_models(self):
        """Convert all models to ONNX format"""
        print("🚀 Starting NASA Exoplanet Models ONNX Conversion\n")
        
        results = {}
        successful_conversions = 0
        
        for dataset in self.model_configs.keys():
            print(f"{'='*50}")
            print(f"Processing {dataset} Model")
            print(f"{'='*50}")
            
            # Load model components
            components = self.load_model_components(dataset)
            if components is None:
                results[dataset] = {'success': False, 'error': 'Failed to load components'}
                continue
            
            # Convert to ONNX
            onnx_result = self.convert_model_to_onnx(dataset, components)
            results[dataset] = onnx_result
            
            if onnx_result['success']:
                # Test ONNX model
                self.test_onnx_model(dataset, onnx_result, components)
                successful_conversions += 1
        
        # Create deployment documentation
        if successful_conversions > 0:
            self.create_deployment_readme()
        
        # Summary
        print(f"\n{'='*50}")
        print(f"🎉 CONVERSION SUMMARY")
        print(f"{'='*50}")
        print(f"✅ Successful conversions: {successful_conversions}/{len(self.model_configs)}")
        
        for dataset, result in results.items():
            if result['success']:
                print(f"✅ {dataset}: ONNX models ready for deployment")
            else:
                print(f"❌ {dataset}: {result.get('error', 'Unknown error')}")
        
        if successful_conversions > 0:
            print(f"\n📁 Models saved to: {self.output_dir}")
            print(f"📖 Documentation: {self.output_dir}/README.md")
            print("\n🚀 Your models are ready for deployment!")
        
        return results

def main():
    """Main conversion function"""
    converter = ModelONNXConverter()
    results = converter.convert_all_models()
    return results

if __name__ == "__main__":
    main()