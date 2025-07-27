#!/usr/bin/env python3
"""
Pre-compute ML results and save as static JSON files
This script runs the full ML pipeline once and saves the results for fast serving
"""

import sys
import json
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the ML processor functions
from ml_processor import (
    load_data, 
    preprocess_data, 
    train_models, 
    hyperparameter_tuning,
    get_dataset_info
)

def main():
    """Pre-compute all ML results and save as JSON files"""
    print("Starting pre-computation of ML results...")
    
    try:
        # Load data
        print("1. Loading 20 newsgroups dataset...")
        newsgroups = load_data()
        
        # Preprocess data
        print("2. Preprocessing data...")
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         vectorizer, feature_names, target_names) = preprocess_data(newsgroups)
        
        # Get dataset info
        print("3. Getting dataset information...")
        dataset_info = get_dataset_info(newsgroups, vectorizer, feature_names)
        
        # Perform hyperparameter tuning
        print("4. Performing hyperparameter tuning...")
        tuning_results = hyperparameter_tuning(X_train, X_val, y_train, y_val, target_names)
        
        # Train models
        print("5. Training models...")
        model_results = train_models(X_train, X_val, X_test, y_train, y_val, y_test, 
                                   target_names, dataset_info, {
            'total_samples': len(newsgroups.data),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'train_percentage': round(len(y_train) / len(newsgroups.data) * 100, 1),
            'val_percentage': round(len(y_val) / len(newsgroups.data) * 100, 1),
            'test_percentage': round(len(y_test) / len(newsgroups.data) * 100, 1)
        }, tuning_results)
        
        # Prepare complete output
        output = {
            'dataset_info': dataset_info,
            'model_results': model_results,
            'target_names': list(target_names),
            'hyperparameter_tuning': tuning_results,
            'data_split_info': {
                'total_samples': len(newsgroups.data),
                'train_samples': len(y_train),
                'val_samples': len(y_val),
                'test_samples': len(y_test),
                'train_percentage': round(len(y_train) / len(newsgroups.data) * 100, 1),
                'val_percentage': round(len(y_val) / len(newsgroups.data) * 100, 1),
                'test_percentage': round(len(y_test) / len(newsgroups.data) * 100, 1)
            },
            'metadata': {
                'computed_at': '2025-01-27T19:00:00Z',
                'version': '1.0.0',
                'description': 'Pre-computed ML results for 20 Newsgroups classification'
            }
        }
        
        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent.parent / 'data'
        data_dir.mkdir(exist_ok=True)
        
        # Save the complete results
        output_file = data_dir / 'ml_results.json'
        print(f"6. Saving results to {output_file}...")
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Also save individual components for modular access
        print("7. Saving individual components...")
        
        # Dataset info
        with open(data_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Model results
        with open(data_dir / 'model_results.json', 'w') as f:
            json.dump(model_results, f, indent=2)
        
        # Hyperparameter tuning
        with open(data_dir / 'hyperparameter_tuning.json', 'w') as f:
            json.dump(tuning_results, f, indent=2)
        
        # Target names
        with open(data_dir / 'target_names.json', 'w') as f:
            json.dump(list(target_names), f, indent=2)
        
        print(f"✅ Pre-computation completed successfully!")
        print(f"📁 Results saved to: {data_dir}")
        print(f"📊 Total models trained: {len(model_results)}")
        print(f"🎯 Models with hyperparameter tuning: {len(tuning_results)}")
        print(f"📈 Best CV score: {max([tuning_results[model]['best_score'] for model in tuning_results])*100:.1f}%")
        
        # Print summary of best parameters
        print("\n🏆 Best Hyperparameters Found:")
        for model_name, tuning_data in tuning_results.items():
            best_score = tuning_data['best_score'] * 100
            print(f"  {model_name}: {best_score:.1f}% - {tuning_data['best_params']}")
        
    except Exception as e:
        print(f"❌ Error during pre-computation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 