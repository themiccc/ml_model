# Machine Learning Model with Scikit-learn

A comprehensive machine learning project that demonstrates model building using Scikit-learn to solve the classic Iris species classification problem.

## 🎯 Problem Statement

Build and compare multiple machine learning models to classify iris flowers into three species (setosa, versicolor, virginica) based on four features:
- Sepal length (cm)
- Sepal width (cm)  
- Petal length (cm)
- Petal width (cm)

## 📁 Project Structure

```
windsurf-project/
├── requirements.txt          # Python dependencies
├── ml_model.py              # Main ML pipeline script
├── README.md                # Project documentation
└── generated_plots/         # Output visualizations (created after running)
    ├── data_exploration.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

## 🛠️ Installation

1. Clone or download the project
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the complete ML pipeline:

```bash
python ml_model.py
```

## 📊 Models Implemented

The project implements and compares three different classification algorithms:

1. **Random Forest Classifier**
   - Ensemble method using multiple decision trees
   - Provides feature importance analysis
   - Robust to overfitting

2. **Support Vector Machine (SVM)**
   - Uses RBF kernel for non-linear classification
   - Effective in high-dimensional spaces
   - Good generalization performance

3. **Logistic Regression**
   - Linear classification algorithm
   - Provides probability estimates
   - Interpretable coefficients

## 🔍 Pipeline Steps

1. **Data Loading**: Load the Iris dataset and create a structured DataFrame
2. **Data Exploration**: Visualize feature distributions and relationships
3. **Data Preprocessing**: Split data and apply feature scaling
4. **Model Training**: Train multiple models with cross-validation
5. **Model Evaluation**: Compare performance using accuracy and other metrics
6. **Visualization**: Generate plots for data exploration and model results
7. **Prediction**: Demonstrate prediction on new samples

## 📈 Evaluation Metrics

- **Test Accuracy**: Performance on unseen test data
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Classification Report**: Precision, Recall, F1-score per class
- **Confusion Matrix**: Detailed classification results
- **Feature Importance**: Relative importance of features (Random Forest)

## 📊 Expected Outputs

When you run the script, you'll get:

1. **Console Output**:
   - Data exploration statistics
   - Model comparison table
   - Best performing model identification
   - Detailed classification report
   - Example prediction results

2. **Visualizations**:
   - Data exploration plots (scatter plots, box plots)
   - Confusion matrix heatmap
   - Feature importance chart

## 🎯 Key Features

- **Modular Design**: Clean class-based architecture for easy extension
- **Multiple Models**: Compare different algorithms automatically
- **Comprehensive Evaluation**: Robust metrics and visualizations
- **Reproducible Results**: Fixed random seeds for consistent outcomes
- **New Predictions**: Function to predict on custom samples

## 🔧 Customization

You can easily extend this project by:

1. **Adding New Models**: Update the `build_models()` method
2. **Different Datasets**: Modify the `load_data()` method
3. **Additional Metrics**: Extend the `evaluate_models()` method
4. **Hyperparameter Tuning**: Add GridSearchCV or RandomizedSearchCV

## 📚 Dependencies

- `scikit-learn`: Machine learning algorithms and utilities
- `numpy`: Numerical computing
- `pandas`: Data manipulation and analysis
- `matplotlib`: Basic plotting
- `seaborn`: Statistical data visualization

## 🏆 Results

The script will automatically identify the best performing model based on test accuracy. For the Iris dataset, you can expect accuracies around 95-98% with all three models performing well.

## 📝 Example Prediction

The script includes an example prediction using sample measurements:
```python
sample = [5.1, 3.5, 1.4, 0.2]  # Typical setosa measurements
```

This demonstrates how to use the trained models for new data predictions.

## 🤝 Contributing

Feel free to modify and extend this project for your learning or specific use cases!
