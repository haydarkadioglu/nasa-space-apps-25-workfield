import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from threading import Thread
import warnings
warnings.filterwarnings('ignore')

class ExoplanetClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 NASA Exoplanet Classification Models")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.models = {}
        self.current_dataset = tk.StringVar(value="K2")
        self.prediction_result = tk.StringVar()
        self.confidence_scores = {}
        
        # Style configuration
        self.setup_styles()
        
        # Load models
        self.load_all_models()
        
        # Create GUI
        self.create_widgets()
        
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Subheader.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Info.TLabel', font=('Arial', 10), background='#f0f0f0')
        style.configure('Success.TLabel', foreground='#28a745', font=('Arial', 12, 'bold'))
        style.configure('Warning.TLabel', foreground='#ffc107', font=('Arial', 12, 'bold'))
        style.configure('Error.TLabel', foreground='#dc3545', font=('Arial', 12, 'bold'))
        
        # Button styles
        style.configure('Primary.TButton', font=('Arial', 11, 'bold'))
        style.configure('Secondary.TButton', font=('Arial', 10))
        
    def load_all_models(self):
        """Load all three models and their metadata"""
        datasets = ["K2", "Kepler", "TESS"]
        
        for dataset in datasets:
            try:
                if dataset == "K2":
                    model_path = "main/k2/k2_3class_best_model.joblib"
                    metadata_path = "main/k2/k2_3class_model_metadata.json"
                    features_path = "main/k2/k2_3class_feature_names.json"
                    scaler_path = "main/k2/k2_3class_scaler.joblib"
                elif dataset == "Kepler":
                    model_path = "main/kepler/kepler_3class_best_model.joblib"
                    metadata_path = "main/kepler/kepler_3class_model_metadata.json"
                    features_path = "main/kepler/kepler_3class_feature_names.json"
                    scaler_path = "main/kepler/kepler_3class_scaler.joblib"
                elif dataset == "TESS":
                    model_path = "tess/tess_models/tess_3class_best_model.joblib"
                    metadata_path = "tess/tess_models/tess_3class_model_metadata.json"
                    features_path = "tess/tess_models/tess_3class_feature_names.json"
                    scaler_path = "tess/tess_models/tess_3class_scaler.joblib"
                
                # Load components
                model = joblib.load(model_path)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                with open(features_path, 'r') as f:
                    features = json.load(f)
                scaler = joblib.load(scaler_path)
                
                self.models[dataset] = {
                    'model': model,
                    'metadata': metadata,
                    'features': features,
                    'scaler': scaler,
                    'status': 'loaded'
                }
                
            except Exception as e:
                self.models[dataset] = {
                    'status': 'error',
                    'error': str(e)
                }
                
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_overview_tab()
        self.create_prediction_tab()
        self.create_comparison_tab()
        self.create_batch_tab()
        
    def create_overview_tab(self):
        """Create model overview tab"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="🏠 Model Overview")
        
        # Header
        header_label = ttk.Label(overview_frame, 
                                text="🚀 NASA Exoplanet Classification Models", 
                                style='Header.TLabel')
        header_label.pack(pady=20)
        
        # Models frame
        models_frame = ttk.Frame(overview_frame)
        models_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create model cards
        for i, (dataset, model_info) in enumerate(self.models.items()):
            self.create_model_card(models_frame, dataset, model_info, i)
        
        # Classification info
        self.create_classification_info(overview_frame)
        
    def create_model_card(self, parent, dataset, model_info, index):
        """Create a model information card"""
        # Model frame
        model_frame = ttk.LabelFrame(parent, text=f"🌌 {dataset} Mission Model", padding=15)
        model_frame.grid(row=index//3, column=index%3, padx=10, pady=10, sticky='nsew')
        
        if model_info['status'] == 'loaded':
            metadata = model_info['metadata']
            
            # Status
            status_label = ttk.Label(model_frame, text="✅ Model Loaded", style='Success.TLabel')
            status_label.pack(anchor='w')
            
            # Metrics
            metrics = [
                f"Model Type: {metadata['model_type']}",
                f"F1-Macro: {metadata['f1_macro_score']:.4f}",
                f"Accuracy: {metadata['accuracy_score']:.4f}",
                f"Training Samples: {metadata['training_samples']:,}",
                f"Features: {metadata['features_count']}",
                f"Training Date: {metadata['training_date']}"
            ]
            
            for metric in metrics:
                metric_label = ttk.Label(model_frame, text=metric, style='Info.TLabel')
                metric_label.pack(anchor='w', pady=2)
                
        else:
            # Error status
            status_label = ttk.Label(model_frame, text="❌ Model Error", style='Error.TLabel')
            status_label.pack(anchor='w')
            
            error_label = ttk.Label(model_frame, text=f"Error: {model_info['error']}", 
                                  style='Info.TLabel', wraplength=200)
            error_label.pack(anchor='w', pady=5)
        
        # Configure grid weights
        parent.grid_columnconfigure(index%3, weight=1)
        
    def create_classification_info(self, parent):
        """Create classification categories info"""
        info_frame = ttk.LabelFrame(parent, text="🎯 Classification Categories", padding=15)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        categories_frame = ttk.Frame(info_frame)
        categories_frame.pack(fill='x')
        
        categories = [
            ("🔍 Candidate", "Planet candidates requiring follow-up observation"),
            ("✅ Confirmed", "Confirmed exoplanets with high confidence"),
            ("❌ False Positive", "False positives and refuted planetary candidates")
        ]
        
        for i, (title, description) in enumerate(categories):
            cat_frame = ttk.Frame(categories_frame)
            cat_frame.grid(row=0, column=i, padx=10, pady=5, sticky='ew')
            
            title_label = ttk.Label(cat_frame, text=title, style='Subheader.TLabel')
            title_label.pack(anchor='w')
            
            desc_label = ttk.Label(cat_frame, text=description, style='Info.TLabel', wraplength=200)
            desc_label.pack(anchor='w')
            
            categories_frame.grid_columnconfigure(i, weight=1)
    
    def create_prediction_tab(self):
        """Create prediction tab"""
        pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(pred_frame, text="🔮 Make Predictions")
        
        # Header
        header_label = ttk.Label(pred_frame, text="🔮 Make Predictions", style='Header.TLabel')
        header_label.pack(pady=20)
        
        # Model selection
        model_frame = ttk.LabelFrame(pred_frame, text="Select Model", padding=10)
        model_frame.pack(fill='x', padx=20, pady=10)
        
        self.model_var = tk.StringVar(value="K2")
        for dataset in ["K2", "Kepler", "TESS"]:
            radio = ttk.Radiobutton(model_frame, text=f"{dataset} Model", 
                                  variable=self.model_var, value=dataset,
                                  command=self.on_model_change)
            radio.pack(side='left', padx=20)
        
        # Current model info
        self.current_model_frame = ttk.LabelFrame(pred_frame, text="Current Model Info", padding=10)
        self.current_model_frame.pack(fill='x', padx=20, pady=10)
        self.update_current_model_info()
        
        # Input methods
        input_frame = ttk.LabelFrame(pred_frame, text="Input Method", padding=10)
        input_frame.pack(fill='x', padx=20, pady=10)
        
        self.input_method = tk.StringVar(value="sample")
        
        ttk.Radiobutton(input_frame, text="🎲 Use Sample Data", 
                       variable=self.input_method, value="sample").pack(anchor='w')
        ttk.Radiobutton(input_frame, text="📊 Upload CSV File", 
                       variable=self.input_method, value="csv").pack(anchor='w')
        ttk.Radiobutton(input_frame, text="✏️ Manual Input", 
                       variable=self.input_method, value="manual").pack(anchor='w')
        
        # Input area
        self.input_area_frame = ttk.Frame(pred_frame)
        self.input_area_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.create_sample_input()
        
        # Control buttons
        control_frame = ttk.Frame(pred_frame)
        control_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(control_frame, text="🔄 Refresh Input", 
                  command=self.refresh_input, style='Secondary.TButton').pack(side='left', padx=5)
        ttk.Button(control_frame, text="🚀 Make Prediction", 
                  command=self.make_prediction, style='Primary.TButton').pack(side='right', padx=5)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(pred_frame, text="🎯 Prediction Results", padding=10)
        self.results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
    def create_sample_input(self):
        """Create sample data input interface"""
        # Clear existing widgets
        for widget in self.input_area_frame.winfo_children():
            widget.destroy()
            
        sample_frame = ttk.LabelFrame(self.input_area_frame, text="Sample Data Selection", padding=10)
        sample_frame.pack(fill='x', pady=5)
        
        # Load sample data
        try:
            dataset = self.model_var.get()
            if dataset == "K2":
                self.sample_data = pd.read_csv("k2/k2.csv").head(10)
            elif dataset == "Kepler":
                self.sample_data = pd.read_csv("kepler/kepler.csv").head(10)
            elif dataset == "TESS":
                self.sample_data = pd.read_csv("tess/TOI.csv").head(10)
            
            ttk.Label(sample_frame, text=f"Sample data loaded: {len(self.sample_data)} rows", 
                     style='Success.TLabel').pack(anchor='w')
            
            # Sample selection
            self.sample_var = tk.IntVar(value=0)
            sample_select_frame = ttk.Frame(sample_frame)
            sample_select_frame.pack(fill='x', pady=5)
            
            ttk.Label(sample_select_frame, text="Select sample:", style='Info.TLabel').pack(side='left')
            sample_combo = ttk.Combobox(sample_select_frame, textvariable=self.sample_var, 
                                       values=list(range(len(self.sample_data))), state='readonly')
            sample_combo.pack(side='left', padx=10)
            
        except Exception as e:
            ttk.Label(sample_frame, text=f"Error loading sample data: {str(e)}", 
                     style='Error.TLabel').pack(anchor='w')
    
    def on_model_change(self):
        """Handle model selection change"""
        self.update_current_model_info()
        self.refresh_input()
        
    def update_current_model_info(self):
        """Update current model information display"""
        # Clear existing widgets
        for widget in self.current_model_frame.winfo_children():
            widget.destroy()
            
        dataset = self.model_var.get()
        model_info = self.models.get(dataset, {})
        
        if model_info.get('status') == 'loaded':
            metadata = model_info['metadata']
            
            info_text = (f"✅ {dataset} Model Loaded | "
                        f"F1-Macro: {metadata['f1_macro_score']:.4f} | "
                        f"Accuracy: {metadata['accuracy_score']:.4f} | "
                        f"Features: {metadata['features_count']}")
            
            ttk.Label(self.current_model_frame, text=info_text, style='Success.TLabel').pack()
        else:
            ttk.Label(self.current_model_frame, text=f"❌ {dataset} Model Error", 
                     style='Error.TLabel').pack()
    
    def refresh_input(self):
        """Refresh input area based on selected method"""
        method = self.input_method.get()
        
        if method == "sample":
            self.create_sample_input()
        elif method == "csv":
            self.create_csv_input()
        elif method == "manual":
            self.create_manual_input()
    
    def create_csv_input(self):
        """Create CSV file input interface"""
        # Clear existing widgets
        for widget in self.input_area_frame.winfo_children():
            widget.destroy()
            
        csv_frame = ttk.LabelFrame(self.input_area_frame, text="CSV File Upload", padding=10)
        csv_frame.pack(fill='both', expand=True, pady=5)
        
        # File selection
        file_frame = ttk.Frame(csv_frame)
        file_frame.pack(fill='x', pady=5)
        
        self.csv_file_path = tk.StringVar()
        ttk.Label(file_frame, text="Select CSV file:", style='Info.TLabel').pack(side='left')
        ttk.Entry(file_frame, textvariable=self.csv_file_path, width=40).pack(side='left', padx=10)
        ttk.Button(file_frame, text="Browse", command=self.browse_csv_file).pack(side='left')
        
        # Data preview area
        self.csv_preview_frame = ttk.Frame(csv_frame)
        self.csv_preview_frame.pack(fill='both', expand=True, pady=10)
        
    def create_manual_input(self):
        """Create manual input interface"""
        # Clear existing widgets
        for widget in self.input_area_frame.winfo_children():
            widget.destroy()
            
        manual_frame = ttk.LabelFrame(self.input_area_frame, text="Manual Feature Input", padding=10)
        manual_frame.pack(fill='both', expand=True, pady=5)
        
        # Key features input
        features_frame = ttk.Frame(manual_frame)
        features_frame.pack(fill='x', pady=5)
        
        # Create input fields for key astronomical features
        self.manual_features = {}
        key_features = [
            ("Orbital Period (days)", "period", 10.0),
            ("Transit Depth (ppm)", "depth", 1000.0),
            ("Transit Duration (hours)", "duration", 3.0),
            ("Stellar Temperature (K)", "temperature", 5500.0),
            ("Stellar Radius (Solar Radii)", "stellar_radius", 1.0),
            ("Stellar Mass (Solar Masses)", "stellar_mass", 1.0)
        ]
        
        for i, (label, key, default) in enumerate(key_features):
            row = i // 2
            col = i % 2
            
            feature_frame = ttk.Frame(features_frame)
            feature_frame.grid(row=row, column=col, padx=10, pady=5, sticky='ew')
            
            ttk.Label(feature_frame, text=label, style='Info.TLabel').pack(anchor='w')
            var = tk.DoubleVar(value=default)
            ttk.Entry(feature_frame, textvariable=var, width=20).pack(fill='x')
            self.manual_features[key] = var
        
        features_frame.grid_columnconfigure(0, weight=1)
        features_frame.grid_columnconfigure(1, weight=1)
        
        # Warning
        warning_label = ttk.Label(manual_frame, 
                                 text="⚠️ Manual input is simplified. For accurate predictions, use sample data or CSV files.",
                                 style='Warning.TLabel')
        warning_label.pack(pady=10)
    
    def browse_csv_file(self):
        """Browse and select CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.csv_file_path.set(file_path)
            self.load_csv_preview(file_path)
    
    def load_csv_preview(self, file_path):
        """Load and preview CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Clear preview frame
            for widget in self.csv_preview_frame.winfo_children():
                widget.destroy()
            
            ttk.Label(self.csv_preview_frame, 
                     text=f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns",
                     style='Success.TLabel').pack(anchor='w')
            
            # Create treeview for data preview
            tree_frame = ttk.Frame(self.csv_preview_frame)
            tree_frame.pack(fill='both', expand=True, pady=5)
            
            # Treeview with scrollbars
            tree = ttk.Treeview(tree_frame)
            v_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
            h_scrollbar = ttk.Scrollbar(tree_frame, orient='horizontal', command=tree.xview)
            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            # Configure columns
            tree['columns'] = list(df.columns)
            tree['show'] = 'headings'
            
            for col in df.columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            
            # Add data (first 10 rows)
            for _, row in df.head(10).iterrows():
                tree.insert('', 'end', values=list(row))
            
            # Pack treeview and scrollbars
            tree.grid(row=0, column=0, sticky='nsew')
            v_scrollbar.grid(row=0, column=1, sticky='ns')
            h_scrollbar.grid(row=1, column=0, sticky='ew')
            
            tree_frame.grid_rowconfigure(0, weight=1)
            tree_frame.grid_columnconfigure(0, weight=1)
            
            self.csv_data = df
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading CSV file: {str(e)}")
    
    def make_prediction(self):
        """Make prediction based on selected input method"""
        dataset = self.model_var.get()
        model_info = self.models.get(dataset)
        
        if not model_info or model_info.get('status') != 'loaded':
            messagebox.showerror("Error", f"{dataset} model is not loaded.")
            return
        
        try:
            method = self.input_method.get()
            
            if method == "sample":
                self.predict_sample()
            elif method == "csv":
                self.predict_csv()
            elif method == "manual":
                self.predict_manual()
                
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction: {str(e)}")
    
    def predict_sample(self):
        """Make prediction on sample data"""
        dataset = self.model_var.get()
        model_info = self.models[dataset]
        
        if not hasattr(self, 'sample_data'):
            messagebox.showerror("Error", "No sample data loaded.")
            return
        
        sample_idx = self.sample_var.get()
        features = model_info['features']
        
        # Extract features for the selected sample
        sample_row = self.sample_data.iloc[sample_idx]
        feature_values = []
        
        for feature in features:
            if feature in sample_row:
                value = sample_row[feature]
                if pd.isna(value):
                    value = 0  # Simple imputation
                feature_values.append(value)
            else:
                feature_values.append(0)  # Default value
        
        # Make prediction
        X_input = np.array(feature_values).reshape(1, -1)
        X_scaled = model_info['scaler'].transform(X_input)
        prediction = model_info['model'].predict(X_scaled)[0]
        probabilities = model_info['model'].predict_proba(X_scaled)[0]
        
        self.display_single_prediction(prediction, probabilities, dataset)
    
    def predict_manual(self):
        """Make prediction on manual input (simplified)"""
        dataset = self.model_var.get()
        
        # This is a simplified demonstration
        messagebox.showwarning("Demo Mode", 
                             "Manual input is in demo mode. Using random probabilities for demonstration.")
        
        # Mock prediction for demonstration
        mock_probabilities = np.random.dirichlet([1, 1, 1])
        mock_prediction = np.argmax(mock_probabilities)
        
        self.display_single_prediction(mock_prediction, mock_probabilities, dataset, is_demo=True)
    
    def predict_csv(self):
        """Make prediction on CSV data"""
        if not hasattr(self, 'csv_data'):
            messagebox.showerror("Error", "No CSV data loaded.")
            return
        
        dataset = self.model_var.get()
        model_info = self.models[dataset]
        features = model_info['features']
        
        predictions = []
        
        for idx, row in self.csv_data.iterrows():
            feature_values = []
            for feature in features:
                if feature in row:
                    value = row[feature]
                    if pd.isna(value):
                        value = 0
                    feature_values.append(value)
                else:
                    feature_values.append(0)
            
            X_input = np.array(feature_values).reshape(1, -1)
            X_scaled = model_info['scaler'].transform(X_input)
            pred = model_info['model'].predict(X_scaled)[0]
            prob = model_info['model'].predict_proba(X_scaled)[0]
            
            class_names = ['Candidate', 'Confirmed', 'False Positive']
            predictions.append({
                'Row': idx,
                'Prediction': class_names[pred],
                'Confidence': f"{prob.max():.3f}",
                'Candidate_Prob': f"{prob[0]:.3f}",
                'Confirmed_Prob': f"{prob[1]:.3f}",
                'FalsePos_Prob': f"{prob[2]:.3f}"
            })
        
        self.display_batch_predictions(predictions, dataset)
    
    def display_single_prediction(self, prediction, probabilities, dataset, is_demo=False):
        """Display single prediction results"""
        # Clear results frame
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        class_names = ['Candidate', 'Confirmed', 'False Positive']
        class_icons = ['🔍', '✅', '❌']
        class_colors = ['#ffd700', '#28a745', '#dc3545']
        
        if is_demo:
            demo_label = ttk.Label(self.results_frame, text="🧪 Demo Prediction", style='Warning.TLabel')
            demo_label.pack(pady=5)
        
        # Main prediction result
        result_text = f"{class_icons[prediction]} Prediction: {class_names[prediction]}"
        result_label = ttk.Label(self.results_frame, text=result_text, style='Subheader.TLabel')
        result_label.pack(pady=10)
        
        # Confidence scores
        conf_frame = ttk.Frame(self.results_frame)
        conf_frame.pack(fill='x', pady=10)
        
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            class_frame = ttk.Frame(conf_frame)
            class_frame.grid(row=0, column=i, padx=10, pady=5, sticky='ew')
            
            ttk.Label(class_frame, text=f"{class_icons[i]} {class_name}", 
                     style='Info.TLabel').pack()
            ttk.Label(class_frame, text=f"{prob:.1%}", 
                     style='Subheader.TLabel').pack()
            
            conf_frame.grid_columnconfigure(i, weight=1)
        
        # Create confidence chart
        self.create_confidence_chart(probabilities, class_names, class_colors)
    
    def display_batch_predictions(self, predictions, dataset):
        """Display batch prediction results"""
        # Create new window for batch results
        results_window = tk.Toplevel(self.root)
        results_window.title(f"🎯 Batch Prediction Results - {dataset}")
        results_window.geometry("800x600")
        
        # Results header
        header_label = ttk.Label(results_window, 
                                text=f"🎯 Batch Prediction Results - {dataset} Model",
                                style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Results summary
        summary_frame = ttk.Frame(results_window)
        summary_frame.pack(fill='x', padx=20, pady=10)
        
        total_predictions = len(predictions)
        pred_counts = {}
        for pred in predictions:
            pred_type = pred['Prediction']
            pred_counts[pred_type] = pred_counts.get(pred_type, 0) + 1
        
        summary_text = f"Total predictions: {total_predictions} | "
        for pred_type, count in pred_counts.items():
            percentage = (count / total_predictions) * 100
            summary_text += f"{pred_type}: {count} ({percentage:.1f}%) | "
        
        ttk.Label(summary_frame, text=summary_text.rstrip(" | "), style='Info.TLabel').pack()
        
        # Results table
        table_frame = ttk.Frame(results_window)
        table_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create treeview
        columns = ['Row', 'Prediction', 'Confidence', 'Candidate_Prob', 'Confirmed_Prob', 'FalsePos_Prob']
        tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col.replace('_', ' '))
            tree.column(col, width=120)
        
        # Add data
        for pred in predictions:
            tree.insert('', 'end', values=[pred[col] for col in columns])
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack tree and scrollbars
        tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Close button
        ttk.Button(results_window, text="Close", 
                  command=results_window.destroy).pack(pady=10)
    
    def create_confidence_chart(self, probabilities, class_names, colors):
        """Create confidence scores chart"""
        chart_frame = ttk.LabelFrame(self.results_frame, text="📊 Confidence Chart", padding=10)
        chart_frame.pack(fill='both', expand=True, pady=10)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(class_names, probabilities, color=colors, alpha=0.7)
        
        # Customize chart
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Confidence Scores')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Embed chart in tkinter
        canvas = FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_comparison_tab(self):
        """Create model comparison tab"""
        comp_frame = ttk.Frame(self.notebook)
        self.notebook.add(comp_frame, text="📊 Model Comparison")
        
        # Header
        header_label = ttk.Label(comp_frame, text="📊 Model Comparison", style='Header.TLabel')
        header_label.pack(pady=20)
        
        # Performance comparison
        perf_frame = ttk.LabelFrame(comp_frame, text="🏆 Performance Comparison", padding=15)
        perf_frame.pack(fill='x', padx=20, pady=10)
        
        # Create performance comparison chart
        self.create_performance_chart(perf_frame)
        
        # Cross-model prediction
        cross_frame = ttk.LabelFrame(comp_frame, text="🔄 Cross-Model Predictions", padding=15)
        cross_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Sample selection for cross-model
        sample_frame = ttk.Frame(cross_frame)
        sample_frame.pack(fill='x', pady=10)
        
        ttk.Label(sample_frame, text="Select dataset for sample:", style='Info.TLabel').pack(side='left')
        self.cross_dataset_var = tk.StringVar(value="K2")
        dataset_combo = ttk.Combobox(sample_frame, textvariable=self.cross_dataset_var,
                                   values=["K2", "Kepler", "TESS"], state='readonly')
        dataset_combo.pack(side='left', padx=10)
        
        self.cross_sample_var = tk.IntVar(value=0)
        ttk.Label(sample_frame, text="Sample index:", style='Info.TLabel').pack(side='left', padx=(20,5))
        sample_spin = ttk.Spinbox(sample_frame, from_=0, to=9, textvariable=self.cross_sample_var, width=5)
        sample_spin.pack(side='left', padx=5)
        
        ttk.Button(sample_frame, text="🚀 Compare All Models", 
                  command=self.compare_all_models, style='Primary.TButton').pack(side='right')
        
        # Results area
        self.cross_results_frame = ttk.Frame(cross_frame)
        self.cross_results_frame.pack(fill='both', expand=True, pady=10)
    
    def create_performance_chart(self, parent):
        """Create performance comparison chart"""
        # Collect performance data
        perf_data = []
        for dataset, model_info in self.models.items():
            if model_info.get('status') == 'loaded':
                metadata = model_info['metadata']
                perf_data.append({
                    'Dataset': dataset,
                    'F1-Macro': metadata['f1_macro_score'],
                    'Accuracy': metadata['accuracy_score'],
                    'Training_Samples': metadata['training_samples']
                })
        
        if not perf_data:
            ttk.Label(parent, text="No performance data available", style='Error.TLabel').pack()
            return
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        datasets = [d['Dataset'] for d in perf_data]
        f1_scores = [d['F1-Macro'] for d in perf_data]
        accuracies = [d['Accuracy'] for d in perf_data]
        samples = [d['Training_Samples'] for d in perf_data]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        # F1-Macro comparison
        bars1 = ax1.bar(datasets, f1_scores, color=colors[:len(datasets)], alpha=0.7)
        ax1.set_ylabel('F1-Macro Score')
        ax1.set_title('F1-Macro Comparison')
        ax1.set_ylim(0, 1)
        
        for bar, score in zip(bars1, f1_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Accuracy comparison
        bars2 = ax2.bar(datasets, accuracies, color=colors[:len(datasets)], alpha=0.7)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Comparison')
        ax2.set_ylim(0, 1)
        
        for bar, acc in zip(bars2, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Embed chart
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def compare_all_models(self):
        """Compare predictions across all models"""
        # Clear results frame
        for widget in self.cross_results_frame.winfo_children():
            widget.destroy()
        
        dataset = self.cross_dataset_var.get()
        sample_idx = self.cross_sample_var.get()
        
        try:
            # Load sample data
            if dataset == "K2":
                sample_data = pd.read_csv("k2/k2.csv").head(10)
            elif dataset == "Kepler":
                sample_data = pd.read_csv("kepler/kepler.csv").head(10)
            elif dataset == "TESS":
                sample_data = pd.read_csv("tess/TOI.csv").head(10)
            
            sample_row = sample_data.iloc[sample_idx]
            
            results = {}
            
            # Get predictions from all available models
            for model_name, model_info in self.models.items():
                if model_info.get('status') == 'loaded':
                    try:
                        features = model_info['features']
                        
                        # Extract features
                        feature_values = []
                        for feature in features:
                            if feature in sample_row:
                                value = sample_row[feature]
                                if pd.isna(value):
                                    value = 0
                                feature_values.append(value)
                            else:
                                feature_values.append(0)
                        
                        # Make prediction
                        X_input = np.array(feature_values).reshape(1, -1)
                        X_scaled = model_info['scaler'].transform(X_input)
                        prediction = model_info['model'].predict(X_scaled)[0]
                        probabilities = model_info['model'].predict_proba(X_scaled)[0]
                        
                        class_names = ['Candidate', 'Confirmed', 'False Positive']
                        results[model_name] = {
                            'prediction': class_names[prediction],
                            'probabilities': probabilities,
                            'confidence': probabilities.max()
                        }
                        
                    except Exception as e:
                        results[model_name] = {'error': str(e)}
            
            if results:
                self.display_cross_results(results, dataset, sample_idx)
            else:
                ttk.Label(self.cross_results_frame, text="No models available for comparison", 
                         style='Error.TLabel').pack()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error in cross-model comparison: {str(e)}")
    
    def display_cross_results(self, results, source_dataset, sample_idx):
        """Display cross-model comparison results"""
        # Results header
        header_text = f"🎯 Cross-Model Results for {source_dataset} Sample #{sample_idx}"
        header_label = ttk.Label(self.cross_results_frame, text=header_text, style='Subheader.TLabel')
        header_label.pack(pady=10)
        
        # Results table
        table_frame = ttk.Frame(self.cross_results_frame)
        table_frame.pack(fill='x', pady=10)
        
        # Headers
        headers = ['Model', 'Prediction', 'Confidence', 'Candidate', 'Confirmed', 'False Positive']
        for i, header in enumerate(headers):
            ttk.Label(table_frame, text=header, style='Subheader.TLabel').grid(row=0, column=i, padx=5, pady=2)
        
        # Results
        row = 1
        valid_results = {}
        
        for model_name, result in results.items():
            if 'error' not in result:
                ttk.Label(table_frame, text=model_name, style='Info.TLabel').grid(row=row, column=0, padx=5, pady=2)
                ttk.Label(table_frame, text=result['prediction'], style='Info.TLabel').grid(row=row, column=1, padx=5, pady=2)
                ttk.Label(table_frame, text=f"{result['confidence']:.1%}", style='Info.TLabel').grid(row=row, column=2, padx=5, pady=2)
                ttk.Label(table_frame, text=f"{result['probabilities'][0]:.1%}", style='Info.TLabel').grid(row=row, column=3, padx=5, pady=2)
                ttk.Label(table_frame, text=f"{result['probabilities'][1]:.1%}", style='Info.TLabel').grid(row=row, column=4, padx=5, pady=2)
                ttk.Label(table_frame, text=f"{result['probabilities'][2]:.1%}", style='Info.TLabel').grid(row=row, column=5, padx=5, pady=2)
                
                valid_results[model_name] = result
                row += 1
        
        # Configure column weights
        for i in range(len(headers)):
            table_frame.grid_columnconfigure(i, weight=1)
        
        # Create comparison chart
        if len(valid_results) > 1:
            self.create_cross_comparison_chart(valid_results)
    
    def create_cross_comparison_chart(self, results):
        """Create cross-model comparison chart"""
        chart_frame = ttk.LabelFrame(self.cross_results_frame, text="📊 Cross-Model Probability Comparison", padding=10)
        chart_frame.pack(fill='both', expand=True, pady=10)
        
        # Prepare data
        models = list(results.keys())
        class_names = ['Candidate', 'Confirmed', 'False Positive']
        colors = ['#ffd700', '#28a745', '#dc3545']
        
        # Create subplots
        fig, axes = plt.subplots(1, len(models), figsize=(4*len(models), 4))
        if len(models) == 1:
            axes = [axes]
        
        for i, (model, result) in enumerate(results.items()):
            ax = axes[i]
            bars = ax.bar(class_names, result['probabilities'], color=colors, alpha=0.7)
            ax.set_title(f'{model} Model')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, prob in zip(bars, result['probabilities']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Embed chart
        canvas = FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_batch_tab(self):
        """Create batch processing tab"""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="📈 Batch Processing")
        
        # Header
        header_label = ttk.Label(batch_frame, text="📈 Batch Processing", style='Header.TLabel')
        header_label.pack(pady=20)
        
        # Instructions
        inst_frame = ttk.LabelFrame(batch_frame, text="Instructions", padding=10)
        inst_frame.pack(fill='x', padx=20, pady=10)
        
        instructions = """
        📋 Batch Processing Instructions:
        1. Select one or more models to use for predictions
        2. Upload a CSV file with astronomical features
        3. Click 'Process Batch' to get predictions from all selected models
        4. Results will be saved to separate CSV files for each model
        """
        
        ttk.Label(inst_frame, text=instructions, style='Info.TLabel', justify='left').pack(anchor='w')
        
        # Model selection
        model_sel_frame = ttk.LabelFrame(batch_frame, text="Select Models", padding=10)
        model_sel_frame.pack(fill='x', padx=20, pady=10)
        
        self.batch_models = {}
        for dataset in ["K2", "Kepler", "TESS"]:
            var = tk.BooleanVar()
            if self.models[dataset].get('status') == 'loaded':
                var.set(True)
                ttk.Checkbutton(model_sel_frame, text=f"{dataset} Model", variable=var).pack(side='left', padx=20)
            else:
                ttk.Checkbutton(model_sel_frame, text=f"{dataset} Model (Not Available)", 
                              variable=var, state='disabled').pack(side='left', padx=20)
            self.batch_models[dataset] = var
        
        # File selection
        file_sel_frame = ttk.LabelFrame(batch_frame, text="Select Input File", padding=10)
        file_sel_frame.pack(fill='x', padx=20, pady=10)
        
        self.batch_file_path = tk.StringVar()
        file_frame = ttk.Frame(file_sel_frame)
        file_frame.pack(fill='x')
        
        ttk.Label(file_frame, text="CSV File:", style='Info.TLabel').pack(side='left')
        ttk.Entry(file_frame, textvariable=self.batch_file_path, width=50).pack(side='left', padx=10)
        ttk.Button(file_frame, text="Browse", command=self.browse_batch_file).pack(side='left')
        
        # Process button
        process_frame = ttk.Frame(batch_frame)
        process_frame.pack(fill='x', padx=20, pady=20)
        
        ttk.Button(process_frame, text="🚀 Process Batch", 
                  command=self.process_batch, style='Primary.TButton').pack()
        
        # Results area
        self.batch_results_frame = ttk.LabelFrame(batch_frame, text="📊 Processing Results", padding=10)
        self.batch_results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Progress bar
        self.batch_progress = ttk.Progressbar(self.batch_results_frame, mode='indeterminate')
        self.batch_progress.pack(fill='x', pady=10)
        
        # Results text
        self.batch_results_text = scrolledtext.ScrolledText(self.batch_results_frame, height=10)
        self.batch_results_text.pack(fill='both', expand=True, pady=10)
    
    def browse_batch_file(self):
        """Browse for batch processing file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV file for batch processing",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.batch_file_path.set(file_path)
    
    def process_batch(self):
        """Process batch file with selected models"""
        # Validate inputs
        selected_models = [dataset for dataset, var in self.batch_models.items() 
                          if var.get() and self.models[dataset].get('status') == 'loaded']
        
        if not selected_models:
            messagebox.showerror("Error", "Please select at least one available model.")
            return
        
        file_path = self.batch_file_path.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a CSV file.")
            return
        
        # Clear results
        self.batch_results_text.delete(1.0, tk.END)
        
        # Start processing in a separate thread
        thread = Thread(target=self.run_batch_processing, args=(selected_models, file_path))
        thread.daemon = True
        thread.start()
        
        # Start progress bar
        self.batch_progress.start()
    
    def run_batch_processing(self, selected_models, file_path):
        """Run batch processing in separate thread"""
        try:
            # Load data
            self.log_batch(f"📂 Loading data from {file_path}...")
            df = pd.read_csv(file_path)
            self.log_batch(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Process each model
            for dataset in selected_models:
                self.log_batch(f"\n🔄 Processing with {dataset} model...")
                
                model_info = self.models[dataset]
                features = model_info['features']
                
                predictions = []
                
                for idx, row in df.iterrows():
                    # Extract features
                    feature_values = []
                    for feature in features:
                        if feature in row:
                            value = row[feature]
                            if pd.isna(value):
                                value = 0
                            feature_values.append(value)
                        else:
                            feature_values.append(0)
                    
                    # Make prediction
                    X_input = np.array(feature_values).reshape(1, -1)
                    X_scaled = model_info['scaler'].transform(X_input)
                    pred = model_info['model'].predict(X_scaled)[0]
                    prob = model_info['model'].predict_proba(X_scaled)[0]
                    
                    class_names = ['Candidate', 'Confirmed', 'False_Positive']
                    predictions.append({
                        'Row_Index': idx,
                        'Prediction': class_names[pred],
                        'Confidence': prob.max(),
                        'Candidate_Probability': prob[0],
                        'Confirmed_Probability': prob[1],
                        'False_Positive_Probability': prob[2]
                    })
                
                # Save results
                output_file = f"{dataset.lower()}_batch_predictions.csv"
                results_df = pd.DataFrame(predictions)
                results_df.to_csv(output_file, index=False)
                
                self.log_batch(f"✅ {dataset} predictions saved to {output_file}")
                
                # Log summary
                pred_counts = results_df['Prediction'].value_counts()
                self.log_batch(f"📊 {dataset} Summary:")
                for pred_type, count in pred_counts.items():
                    percentage = (count / len(results_df)) * 100
                    self.log_batch(f"   • {pred_type}: {count} ({percentage:.1f}%)")
            
            self.log_batch(f"\n🎉 Batch processing completed successfully!")
            
        except Exception as e:
            self.log_batch(f"\n❌ Error during batch processing: {str(e)}")
        
        finally:
            # Stop progress bar
            self.root.after(0, self.batch_progress.stop)
    
    def log_batch(self, message):
        """Log message to batch results text area"""
        def update_text():
            self.batch_results_text.insert(tk.END, message + "\n")
            self.batch_results_text.see(tk.END)
        
        self.root.after(0, update_text)

def main():
    root = tk.Tk()
    app = ExoplanetClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()