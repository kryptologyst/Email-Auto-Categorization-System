#!/usr/bin/env python3
"""
Main entry point for the Email Auto-Categorization System.
Provides both CLI and programmatic interfaces.
"""

import argparse
import sys
from pathlib import Path
from src.database import EmailDatabase
from src.models import EmailClassifier, EnsembleEmailClassifier
from src.visualization import EmailVisualization
import pandas as pd

def setup_database():
    """Initialize database with sample data."""
    print("Setting up database...")
    db = EmailDatabase()
    db.add_sample_data()
    print("✅ Database initialized with sample data")
    return db

def train_model(model_type="ensemble", save_model=True):
    """Train a model and optionally save it."""
    print(f"Training {model_type} model...")
    
    db = EmailDatabase()
    df = db.get_training_emails()
    
    if model_type == "ensemble":
        model = EnsembleEmailClassifier()
    else:
        model = EmailClassifier(model_type)
    
    results = model.train(df)
    
    print(f"✅ Model trained successfully!")
    accuracy = results.get('accuracy', 'N/A')
    if accuracy != 'N/A':
        print(f"   Accuracy: {accuracy:.3f}")
    else:
        print(f"   Accuracy: {accuracy}")
    if 'cv_mean' in results:
        print(f"   CV Score: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    if save_model:
        Path("models").mkdir(exist_ok=True)
        model_path = f"models/{model_type}_model.joblib"
        model.save_model(model_path)
        print(f"💾 Model saved to {model_path}")
    
    return model, results

def predict_email(subject, body, model_type="ensemble"):
    """Predict email category."""
    model_path = f"models/{model_type}_model.joblib"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}. Training new model...")
        model, _ = train_model(model_type)
    else:
        if model_type == "ensemble":
            model = EnsembleEmailClassifier()
        else:
            model = EmailClassifier(model_type)
        model.load_model(model_path)
        print(f"✅ Loaded model from {model_path}")
    
    category, confidence = model.predict(body, subject)
    
    print(f"\n📧 Email Classification Result:")
    print(f"   Subject: {subject}")
    print(f"   Category: {category}")
    print(f"   Confidence: {confidence:.2%}")
    
    return category, confidence

def compare_models():
    """Compare different model performances."""
    print("Comparing model performances...")
    
    db = EmailDatabase()
    df = db.get_training_emails()
    
    models = ["logistic_regression", "random_forest", "gradient_boosting", "svm"]
    results = {}
    
    for model_type in models:
        print(f"Training {model_type}...")
        try:
            model = EmailClassifier(model_type)
            result = model.train(df)
            results[model_type] = result
            print(f"   ✅ {model_type}: {result['accuracy']:.3f}")
        except Exception as e:
            print(f"   ❌ {model_type}: {str(e)}")
    
    # Display comparison
    print("\n📊 Model Comparison Results:")
    print("-" * 50)
    for model_type, result in results.items():
        print(f"{model_type:20} | Accuracy: {result['accuracy']:.3f} | CV: {result['cv_mean']:.3f}±{result['cv_std']:.3f}")
    
    # Find best model
    if results:
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\n🏆 Best model: {best_model} (Accuracy: {results[best_model]['accuracy']:.3f})")
    
    return results

def run_web_app():
    """Launch the Streamlit web application."""
    import subprocess
    import sys
    
    print("🚀 Launching Streamlit web application...")
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Web application stopped")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Email Auto-Categorization System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Initialize database with sample data')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', default='ensemble', 
                             choices=['ensemble', 'logistic_regression', 'random_forest', 'gradient_boosting', 'svm'],
                             help='Model type to train')
    train_parser.add_argument('--no-save', action='store_true', help='Don\'t save the trained model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict email category')
    predict_parser.add_argument('--subject', required=True, help='Email subject')
    predict_parser.add_argument('--body', required=True, help='Email body')
    predict_parser.add_argument('--model', default='ensemble',
                               choices=['ensemble', 'logistic_regression', 'random_forest', 'gradient_boosting', 'svm'],
                               help='Model type to use')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model performances')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Launch web application')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_database()
    
    elif args.command == 'train':
        train_model(args.model, not args.no_save)
    
    elif args.command == 'predict':
        predict_email(args.subject, args.body, args.model)
    
    elif args.command == 'compare':
        compare_models()
    
    elif args.command == 'web':
        run_web_app()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
