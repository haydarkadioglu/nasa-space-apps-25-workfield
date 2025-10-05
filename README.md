# 🌌 NASA Space Apps Challenge 2025 - Workfield

**Challenge: "A World Away: Hunting for Exoplanets with AI"**

*Developed by **Kozmik Zihinler** (Cosmic Minds) Team*

---

## 🎯 Project Overview

This repository contains the complete development workfield for our NASA Space Apps Challenge 2025 submission. We've built an AI-powered exoplanet classification system that analyzes data from NASA's major space missions (Kepler, K2, and TESS) to identify and classify exoplanet candidates.

## 🚀 Final Deployment

**Live Demo**: [NASA Exoplanet Hunter](https://github.com/haydarkadioglu/exoplanet-hunting)

The production-ready application is deployed in a separate repository for clean distribution and easy access.

## 📁 Repository Structure

```
nasa-space-apps-workfield/
├── 📂 archive/                     # Original raw datasets
│   ├── Astro_Flux_Data.csv
│   ├── Astro_Time_Data.csv
│   └── features_dataset.csv
│
├── 📂 k2/                          # K2 Mission Data Processing
│   ├── k2.csv                      # Processed K2 dataset
│   ├── preprocess-k2-vol-1.ipynb  # Data preprocessing notebook
│   ├── train_v1.ipynb             # Model training notebook
│   ├── 📂 vol1/                   # Training data volume 1
│   ├── 📂 vol1_3class/            # 3-class classification data
│   └── info-txt                   # Dataset information
│
├── 📂 kepler/                      # Kepler Mission Data Processing
│   ├── kepler.csv                  # Processed Kepler dataset
│   ├── preprocess_kepler.ipynb    # Data preprocessing notebook
│   ├── train_v1.ipynb             # Model training notebook
│   ├── 📂 kepler_3class/          # 3-class classification data
│   ├── 📂 kepler_models/          # Trained model artifacts
│   └── info.txt                   # Dataset information
│
├── 📂 tess/                        # TESS Mission Data Processing
│   ├── TOI.csv                     # TESS Objects of Interest dataset
│   ├── TIC_IDs.csv                # TESS Input Catalog IDs
│   ├── preprocess_tess.ipynb      # Data preprocessing notebook
│   ├── train_v2.ipynb             # Model training v2
│   ├── train_v3.ipynb             # Model training v3
│   ├── train_v4.ipynb             # Model training v4
│   ├── train_v5.ipynb             # Model training v5
│   ├── v6.ipynb                   # Final model version
│   ├── train_v4_final_comparison.py # Model comparison script
│   ├── 📂 tess_3class/            # 3-class classification data
│   ├── 📂 tess_models/            # Trained model artifacts
│   ├── 📂 autogluon_models/       # AutoGluon experiment results
│   ├── 📂 vol2/ vol3/ vol4/       # Training data volumes
│   ├── 📂 cache/                  # Cached preprocessing results
│   ├── 📂 playground/             # Experimental notebooks
│   ├── toi_feature_mapping.json   # Feature mapping configuration
│   ├── toi_pipeline_metadata.json # Pipeline metadata
│   ├── toi_preprocessing_pipeline.joblib # Preprocessing pipeline
│   └── info.txt                   # Dataset information
│
├── 📂 main/                        # Final model artifacts
│   ├── 📂 k2/                     # K2 production models
│   ├── 📂 kepler/                 # Kepler production models
│   └── 📂 toi/                    # TESS production models
│
├── 📂 csa/                         # Canadian Space Agency data
│   ├── download.py                 # Data download script
│   └── 📂 data/                   # FITS data files
│
├── 📂 exoplanet-hunting/          # Frontend Application
│   ├── 📂 frontend/               # Web application
│   │   ├── 📂 src/                # TypeScript source code
│   │   ├── 📂 public/             # Static assets & ONNX models
│   │   ├── package.json           # Dependencies
│   │   ├── vite.config.ts         # Build configuration
│   │   └── index.html             # Main HTML file
│   └── README.md                  # Application documentation
│
├── 🔧 Utility Scripts
│   ├── convert_to_onnx.py         # Model conversion to ONNX format
│   ├── convert_kepler_special.py  # Special Kepler data conversion
│   ├── tkinter_app.py             # Desktop GUI application
│   └── dataset_comparison_analysis.ipynb # Dataset analysis
│
├── 📊 Data Files
│   ├── tois.csv                   # TESS Objects of Interest
│   ├── archive.zip                # Compressed archive data
│   └── 📂 flux_data/              # Light curve flux data
│
└── 📋 Documentation
    ├── README.md                   # This file
    └── ONNX_CONVERSION_SUMMARY.md  # ONNX conversion documentation
```

## 🛠️ Technical Architecture

### Data Processing Pipeline

1. **Data Ingestion**: Raw astronomical data from NASA archives
2. **Feature Engineering**: Extract meaningful parameters from light curves
3. **Preprocessing**: Normalization, cleaning, and feature selection
4. **Model Training**: Machine learning model development
5. **Model Conversion**: ONNX format for web deployment
6. **Web Integration**: TypeScript frontend with ONNX Runtime

### Machine Learning Models

| Dataset | Algorithm | Features | Accuracy | Status |
|---------|-----------|----------|----------|---------|
| **K2** | Gradient Boosting | 145 | 97.00% | ✅ Production |
| **Kepler** | LightGBM | 106 | 85.36% | ✅ Production |
| **TESS** | Random Forest | 43 | 89.94% | ✅ Production |

### Technology Stack

#### Backend & ML
- **Python 3.8+**: Core development language
- **Scikit-learn**: Machine learning algorithms
- **LightGBM**: Gradient boosting framework
- **AutoGluon**: Automated ML experiments
- **ONNX**: Model serialization and deployment
- **Pandas/NumPy**: Data manipulation
- **Jupyter**: Interactive development

#### Frontend
- **TypeScript**: Type-safe JavaScript development
- **Vite**: Modern build tool and dev server
- **ONNX Runtime Web**: Client-side model inference
- **HTML5/CSS3**: Modern web standards
- **WebAssembly**: High-performance computation

#### Data Sources
- **NASA Exoplanet Archive**: Kepler and K2 mission data
- **MAST Portal**: TESS mission data and TOI catalog
- **Canadian Space Agency**: NEO surveillance data

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with Jupyter
- Node.js 16+ for frontend development
- Git for version control

### Setting Up Development Environment

1. **Clone the repository**:
```bash
git clone https://github.com/haydarkadioglu/nasa-space-apps-25-workfield.git
cd nasa-space-apps-25-workfield
```

2. **Set up Python environment**:
```bash
pip install jupyter pandas numpy scikit-learn lightgbm onnx onnxruntime
```

3. **Explore the data processing**:
```bash
# Start with K2 data processing
jupyter notebook k2/preprocess-k2-vol-1.ipynb

# Or explore TESS data
jupyter notebook tess/preprocess_tess.ipynb
```

4. **Run the web application**:
```bash
cd exoplanet-hunting/frontend
npm install
npm run dev
```

## 📊 Development Workflow

### 1. Data Processing Phase
- **Notebooks**: `*/preprocess_*.ipynb` - Data cleaning and feature engineering
- **Training**: `*/train_*.ipynb` - Model development and evaluation
- **Evaluation**: Confusion matrices and performance metrics

### 2. Model Development Phase
- **Experimentation**: Multiple algorithm testing (Random Forest, LightGBM, etc.)
- **Hyperparameter Tuning**: Grid search and cross-validation
- **Feature Selection**: Importance analysis and dimensionality reduction

### 3. Production Phase
- **ONNX Conversion**: `convert_to_onnx.py` - Model serialization
- **Web Integration**: TypeScript frontend development
- **Performance Optimization**: Client-side inference optimization

## 🔬 Scientific Methodology

### Classification Categories
- **🔍 Candidate**: Transit-like signals requiring verification
- **✅ Confirmed**: Validated exoplanets with high confidence
- **❌ False Positive**: Stellar activity or instrumental artifacts

### Feature Engineering
- **Kepler**: Orbital parameters, stellar characteristics, signal properties
- **K2**: Extended mission features with advanced noise filtering
- **TESS**: TOI catalog parameters optimized for all-sky survey data

### Model Validation
- **Cross-validation**: 5-fold stratified validation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Analysis**: Detailed error analysis and bias detection

## 🏆 Team: Kozmik Zihinler (Cosmic Minds)

A passionate team of developers and data scientists dedicated to advancing exoplanet discovery through artificial intelligence for the **2025 NASA Space Apps Challenge**.

## 📈 Results and Impact

Our AI models demonstrate state-of-the-art performance in exoplanet classification:
- **High Accuracy**: Up to 97% classification accuracy
- **Real-time Inference**: Sub-second prediction times
- **Web Accessibility**: Browser-based tool for researchers worldwide
- **Multi-mission Support**: Unified interface for diverse datasets

## 🔗 Links

- **🌟 Live Application**: [exoplanet-hunting](https://github.com/haydarkadioglu/exoplanet-hunting)
- **📊 NASA Exoplanet Archive**: [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu/)
- **🔭 TESS Mission**: [tess.mit.edu](https://tess.mit.edu/)
- **🛰️ Kepler Mission**: [www.nasa.gov/kepler](https://www.nasa.gov/kepler)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests, report issues, or suggest improvements.

---

*"Advancing the frontier of exoplanet discovery through the power of artificial intelligence."*  
**- Kozmik Zihinler (Cosmic Minds) Team**
