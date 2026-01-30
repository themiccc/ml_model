"""
Machine Learning Model using Scikit-learn
Basic Classification Problem: Iris Dataset Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MLModelBuilder:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare the Iris dataset"""
        print("Loading Iris dataset...")
        iris = load_iris()
        self.data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                                columns=iris['feature_names'] + ['target'])
        
        # Add target names for better understanding
        target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        self.data['species'] = self.data['target'].map(target_names)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {iris['feature_names']}")
        print(f"Target classes: {iris['target_names']}")
        return self.data
    
    def explore_data(self):
        """Perform basic data exploration"""
        print("\n=== Data Exploration ===")
        print(f"Missing values:\n{self.data.isnull().sum()}")
        print(f"\nData description:\n{self.data.describe()}")
        print(f"\nClass distribution:\n{self.data['species'].value_counts()}")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Pair plot
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=self.data, x='sepal length (cm)', y='sepal width (cm)', 
                       hue='species', style='species', s=100)
        plt.title('Sepal Length vs Sepal Width')
        
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=self.data, x='petal length (cm)', y='petal width (cm)', 
                       hue='species', style='species', s=100)
        plt.title('Petal Length vs Petal Width')
        
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.data, x='species', y='sepal length (cm)')
        plt.title('Sepal Length by Species')
        
        plt.subplot(2, 2, 4)
        sns.boxplot(data=self.data, x='species', y='petal length (cm)')
        plt.title('Petal Length by Species')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Split and preprocess the data"""
        print("\n=== Data Preprocessing ===")
        
        # Separate features and target
        X = self.data.drop(['target', 'species'], axis=1)
        y = self.data['target']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preprocessing completed!")
        
    def build_models(self):
        """Build and train multiple models"""
        print("\n=== Building Models ===")
        
        # Initialize models
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train models
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"{name} - Test Accuracy: {accuracy:.4f}, CV Mean: {cv_scores.mean():.4f}")
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n=== Model Evaluation ===")
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test Accuracy': [self.results[name]['accuracy'] for name in self.results.keys()],
            'CV Mean': [self.results[name]['cv_mean'] for name in self.results.keys()],
            'CV Std': [self.results[name]['cv_std'] for name in self.results.keys()]
        })
        
        print(comparison_df.round(4))
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['accuracy'])
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        
        # Detailed evaluation for best model
        best_model = self.results[best_model_name]['model']
        y_pred = self.results[best_model_name]['predictions']
        
        print(f"\n=== Detailed Evaluation - {best_model_name} ===")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['setosa', 'versicolor', 'virginica']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['setosa', 'versicolor', 'virginica'],
                   yticklabels=['setosa', 'versicolor', 'virginica'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_model_name
    
    def feature_importance(self, model_name):
        """Analyze feature importance for tree-based models"""
        if model_name == 'Random Forest':
            model = self.models['Random Forest']
            feature_names = self.data.drop(['target', 'species'], axis=1).columns
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance - Random Forest')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\nFeature Importance:")
            for i, idx in enumerate(indices):
                print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    def predict_new_sample(self, model_name, sample_features):
        """Make prediction on new data"""
        model = self.models[model_name]
        sample_scaled = self.scaler.transform([sample_features])
        prediction = model.predict(sample_scaled)
        prediction_proba = model.predict_proba(sample_scaled)
        
        target_names = ['setosa', 'versicolor', 'virginica']
        predicted_class = target_names[prediction[0]]
        
        print(f"\n=== Prediction for New Sample ===")
        print(f"Features: {sample_features}")
        print(f"Predicted class: {predicted_class}")
        print(f"Probabilities: {dict(zip(target_names, prediction_proba[0]))}")
        
        return predicted_class, prediction_proba[0]

def main():
    """Main function to run the ML pipeline"""
    print("=== Machine Learning Model Building with Scikit-learn ===")
    print("Problem: Iris Species Classification")
    print("=" * 50)
    
    # Initialize the ML builder
    ml_builder = MLModelBuilder()
    
    # Load and explore data
    ml_builder.load_data()
    ml_builder.explore_data()
    
    # Preprocess data
    ml_builder.preprocess_data()
    
    # Build and evaluate models
    ml_builder.build_models()
    best_model = ml_builder.evaluate_models()
    
    # Feature importance analysis
    ml_builder.feature_importance('Random Forest')
    
    # Example prediction
    sample = [5.1, 3.5, 1.4, 0.2]  # Typical setosa measurements
    ml_builder.predict_new_sample(best_model, sample)
    
    print("\n=== ML Pipeline Completed Successfully! ===")

if __name__ == "__main__":
    main()
