#!/usr/bin/env python3
"""
Generate comprehensive sample static data for ML results
This creates realistic sample data without running the full ML pipeline
"""

import json
import os
from pathlib import Path
import random

def generate_confusion_matrix(target_names):
    """Generate realistic confusion matrix data"""
    n_classes = len(target_names)
    matrix = []
    for i in range(n_classes):
        row = []
        for j in range(n_classes):
            if i == j:
                # Diagonal elements (correct predictions) should be higher
                row.append(random.randint(80, 95))
            else:
                # Off-diagonal elements (incorrect predictions) should be lower
                row.append(random.randint(0, 15))
        matrix.append(row)
    return matrix

def generate_roc_data(n_classes):
    """Generate realistic ROC curve data"""
    roc_data = {}
    for i in range(n_classes):
        # Generate realistic FPR and TPR points
        fpr = [0.0] + [random.uniform(0, 1) for _ in range(48)] + [1.0]
        fpr.sort()
        
        # TPR should generally increase with FPR for realistic ROC curves
        tpr = [0.0]
        for j in range(1, 49):
            # TPR should be higher than FPR for good classifiers
            tpr.append(max(fpr[j], random.uniform(fpr[j], min(1.0, fpr[j] + 0.3))))
        tpr.append(1.0)
        
        roc_data[str(i)] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': random.uniform(0.75, 0.98)
        }
    return roc_data

def generate_feature_importance(n_features=100):
    """Generate realistic feature importance scores"""
    features = []
    for i in range(n_features):
        features.append({
            'feature': f'feature_{i:04d}',
            'importance': random.uniform(0.001, 0.05)
        })
    # Sort by importance descending
    features.sort(key=lambda x: x['importance'], reverse=True)
    return features[:50]  # Return top 50 features

def generate_sample_data():
    """Generate comprehensive realistic sample ML results data"""
    
    # Create data directory
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Sample target names (20 newsgroups)
    target_names = [
        'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
        'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
        'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
        'sci.space', 'soc.religion.christian', 'talk.politics.guns',
        'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
    ]
    
    # Generate confusion matrix for all models
    confusion_matrix = generate_confusion_matrix(target_names)
    
    # Generate ROC data
    roc_data = generate_roc_data(len(target_names))
    
    # Generate feature importance
    feature_importance = generate_feature_importance()
    
    # Sample model results for all 10 algorithms with comprehensive data
    model_results = {
        'Logistic Regression': {
            'accuracy': 0.847,
            'precision': 0.851,
            'recall': 0.847,
            'f1_score': 0.848,
            'cv_mean': 0.843,
            'cv_std': 0.012,
            'validation_accuracy': 0.845,
            'predictions': [random.randint(0, 19) for _ in range(3770)],  # Full test set
            'true_labels': [random.randint(0, 19) for _ in range(3770)],
            'confusion_matrix': confusion_matrix,
            'roc_data': roc_data,
            'feature_importance': feature_importance,
            'training_time': 12.5,
            'prediction_time': 0.8,
            'model_size_mb': 2.3
        },
        'Random Forest': {
            'accuracy': 0.892,
            'precision': 0.895,
            'recall': 0.892,
            'f1_score': 0.893,
            'cv_mean': 0.889,
            'cv_std': 0.008,
            'validation_accuracy': 0.891,
            'predictions': [random.randint(0, 19) for _ in range(3770)],
            'true_labels': [random.randint(0, 19) for _ in range(3770)],
            'confusion_matrix': generate_confusion_matrix(target_names),
            'roc_data': generate_roc_data(len(target_names)),
            'feature_importance': generate_feature_importance(),
            'training_time': 45.2,
            'prediction_time': 1.2,
            'model_size_mb': 8.7
        },
        'Support Vector Machine': {
            'accuracy': 0.823,
            'precision': 0.827,
            'recall': 0.823,
            'f1_score': 0.824,
            'cv_mean': 0.819,
            'cv_std': 0.015,
            'validation_accuracy': 0.821,
            'predictions': [random.randint(0, 19) for _ in range(3770)],
            'true_labels': [random.randint(0, 19) for _ in range(3770)],
            'confusion_matrix': generate_confusion_matrix(target_names),
            'roc_data': generate_roc_data(len(target_names)),
            'feature_importance': generate_feature_importance(),
            'training_time': 78.3,
            'prediction_time': 2.1,
            'model_size_mb': 15.2
        },
        'XGBoost': {
            'accuracy': 0.915,
            'precision': 0.918,
            'recall': 0.915,
            'f1_score': 0.916,
            'cv_mean': 0.912,
            'cv_std': 0.006,
            'validation_accuracy': 0.914,
            'predictions': [random.randint(0, 19) for _ in range(3770)],
            'true_labels': [random.randint(0, 19) for _ in range(3770)],
            'confusion_matrix': generate_confusion_matrix(target_names),
            'roc_data': generate_roc_data(len(target_names)),
            'feature_importance': generate_feature_importance(),
            'training_time': 34.7,
            'prediction_time': 0.9,
            'model_size_mb': 6.8
        },
        'K-Nearest Neighbors': {
            'accuracy': 0.756,
            'precision': 0.761,
            'recall': 0.756,
            'f1_score': 0.758,
            'cv_mean': 0.752,
            'cv_std': 0.018,
            'validation_accuracy': 0.754,
            'predictions': [random.randint(0, 19) for _ in range(3770)],
            'true_labels': [random.randint(0, 19) for _ in range(3770)],
            'confusion_matrix': generate_confusion_matrix(target_names),
            'roc_data': generate_roc_data(len(target_names)),
            'feature_importance': generate_feature_importance(),
            'training_time': 2.1,
            'prediction_time': 15.3,
            'model_size_mb': 45.6
        },
        'Naive Bayes': {
            'accuracy': 0.734,
            'precision': 0.738,
            'recall': 0.734,
            'f1_score': 0.736,
            'cv_mean': 0.729,
            'cv_std': 0.021,
            'validation_accuracy': 0.731,
            'predictions': [random.randint(0, 19) for _ in range(3770)],
            'true_labels': [random.randint(0, 19) for _ in range(3770)],
            'confusion_matrix': generate_confusion_matrix(target_names),
            'roc_data': generate_roc_data(len(target_names)),
            'feature_importance': generate_feature_importance(),
            'training_time': 8.9,
            'prediction_time': 0.3,
            'model_size_mb': 1.2
        },
        'Decision Tree': {
            'accuracy': 0.687,
            'precision': 0.692,
            'recall': 0.687,
            'f1_score': 0.689,
            'cv_mean': 0.682,
            'cv_std': 0.025,
            'validation_accuracy': 0.684,
            'predictions': [random.randint(0, 19) for _ in range(3770)],
            'true_labels': [random.randint(0, 19) for _ in range(3770)],
            'confusion_matrix': generate_confusion_matrix(target_names),
            'roc_data': generate_roc_data(len(target_names)),
            'feature_importance': generate_feature_importance(),
            'training_time': 15.6,
            'prediction_time': 0.2,
            'model_size_mb': 3.4
        },
        'AdaBoost': {
            'accuracy': 0.798,
            'precision': 0.802,
            'recall': 0.798,
            'f1_score': 0.800,
            'cv_mean': 0.794,
            'cv_std': 0.016,
            'validation_accuracy': 0.796,
            'predictions': [random.randint(0, 19) for _ in range(3770)],
            'true_labels': [random.randint(0, 19) for _ in range(3770)],
            'confusion_matrix': generate_confusion_matrix(target_names),
            'roc_data': generate_roc_data(len(target_names)),
            'feature_importance': generate_feature_importance(),
            'training_time': 28.4,
            'prediction_time': 0.7,
            'model_size_mb': 4.1
        },
        'Gradient Boosting': {
            'accuracy': 0.903,
            'precision': 0.906,
            'recall': 0.903,
            'f1_score': 0.904,
            'cv_mean': 0.899,
            'cv_std': 0.009,
            'validation_accuracy': 0.901,
            'predictions': [random.randint(0, 19) for _ in range(3770)],
            'true_labels': [random.randint(0, 19) for _ in range(3770)],
            'confusion_matrix': generate_confusion_matrix(target_names),
            'roc_data': generate_roc_data(len(target_names)),
            'feature_importance': generate_feature_importance(),
            'training_time': 52.8,
            'prediction_time': 1.1,
            'model_size_mb': 7.9
        },
        'LightGBM': {
            'accuracy': 0.921,
            'precision': 0.924,
            'recall': 0.921,
            'f1_score': 0.922,
            'cv_mean': 0.918,
            'cv_std': 0.005,
            'validation_accuracy': 0.920,
            'predictions': [random.randint(0, 19) for _ in range(3770)],
            'true_labels': [random.randint(0, 19) for _ in range(3770)],
            'confusion_matrix': generate_confusion_matrix(target_names),
            'roc_data': generate_roc_data(len(target_names)),
            'feature_importance': generate_feature_importance(),
            'training_time': 23.1,
            'prediction_time': 0.6,
            'model_size_mb': 5.3
        }
    }
    
    # Sample hyperparameter tuning results for all 10 algorithms
    hyperparameter_tuning = {
        'Logistic Regression': {
            'best_params': {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000},
            'best_score': 0.847,
            'cv_results': {
                'mean_test_score': [0.823, 0.847, 0.831, 0.839, 0.845, 0.841],
                'std_test_score': [0.015, 0.012, 0.014, 0.013, 0.011, 0.012],
                'params': [
                    {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000},
                    {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000},
                    {'C': 5.0, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000}
                ]
            }
        },
        'Random Forest': {
            'best_params': {'n_estimators': 50, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
            'best_score': 0.892,
            'cv_results': {
                'mean_test_score': [0.876, 0.892, 0.889, 0.885, 0.891, 0.887],
                'std_test_score': [0.010, 0.008, 0.009, 0.011, 0.008, 0.010],
                'params': [
                    {'n_estimators': 25, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
                    {'n_estimators': 50, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'}
                ]
            }
        },
        'Support Vector Machine': {
            'best_params': {'C': 1.0, 'loss': 'squared_hinge', 'max_iter': 1000},
            'best_score': 0.823,
            'cv_results': {
                'mean_test_score': [0.815, 0.823, 0.819, 0.821, 0.817, 0.820],
                'std_test_score': [0.016, 0.015, 0.017, 0.015, 0.016, 0.014],
                'params': [
                    {'C': 0.5, 'loss': 'squared_hinge', 'max_iter': 1000},
                    {'C': 1.0, 'loss': 'squared_hinge', 'max_iter': 1000},
                    {'C': 2.0, 'loss': 'squared_hinge', 'max_iter': 1000}
                ]
            }
        },
        'XGBoost': {
            'best_params': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.2, 'subsample': 0.8, 'colsample_bytree': 0.8},
            'best_score': 0.915,
            'cv_results': {
                'mean_test_score': [0.901, 0.915, 0.908, 0.912, 0.910, 0.913],
                'std_test_score': [0.008, 0.006, 0.007, 0.006, 0.008, 0.007],
                'params': [
                    {'n_estimators': 25, 'max_depth': 2, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
                    {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.2, 'subsample': 0.8, 'colsample_bytree': 0.8}
                ]
            }
        },
        'K-Nearest Neighbors': {
            'best_params': {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'minkowski'},
            'best_score': 0.756,
            'cv_results': {
                'mean_test_score': [0.742, 0.756, 0.748, 0.751, 0.753, 0.749],
                'std_test_score': [0.019, 0.018, 0.020, 0.018, 0.017, 0.019],
                'params': [
                    {'n_neighbors': 3, 'weights': 'uniform', 'metric': 'minkowski'},
                    {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'minkowski'},
                    {'n_neighbors': 7, 'weights': 'uniform', 'metric': 'minkowski'}
                ]
            }
        },
        'Naive Bayes': {
            'best_params': {'alpha': 0.5, 'fit_prior': True},
            'best_score': 0.734,
            'cv_results': {
                'mean_test_score': [0.721, 0.734, 0.728, 0.731, 0.729, 0.732],
                'std_test_score': [0.022, 0.021, 0.023, 0.020, 0.022, 0.021],
                'params': [
                    {'alpha': 0.1, 'fit_prior': True},
                    {'alpha': 0.5, 'fit_prior': True},
                    {'alpha': 1.0, 'fit_prior': True}
                ]
            }
        },
        'Decision Tree': {
            'best_params': {'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'gini'},
            'best_score': 0.687,
            'cv_results': {
                'mean_test_score': [0.672, 0.687, 0.681, 0.684, 0.679, 0.682],
                'std_test_score': [0.026, 0.025, 0.027, 0.024, 0.026, 0.025],
                'params': [
                    {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
                    {'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'gini'},
                    {'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 4, 'criterion': 'gini'}
                ]
            }
        },
        'AdaBoost': {
            'best_params': {'n_estimators': 50, 'learning_rate': 0.5, 'algorithm': 'SAMME'},
            'best_score': 0.798,
            'cv_results': {
                'mean_test_score': [0.785, 0.798, 0.792, 0.795, 0.790, 0.793],
                'std_test_score': [0.017, 0.016, 0.018, 0.015, 0.017, 0.016],
                'params': [
                    {'n_estimators': 25, 'learning_rate': 0.3, 'algorithm': 'SAMME'},
                    {'n_estimators': 50, 'learning_rate': 0.5, 'algorithm': 'SAMME'},
                    {'n_estimators': 75, 'learning_rate': 0.7, 'algorithm': 'SAMME'}
                ]
            }
        },
        'Gradient Boosting': {
            'best_params': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8},
            'best_score': 0.903,
            'cv_results': {
                'mean_test_score': [0.889, 0.903, 0.896, 0.899, 0.894, 0.897],
                'std_test_score': [0.011, 0.009, 0.010, 0.008, 0.011, 0.009],
                'params': [
                    {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.1, 'subsample': 0.8},
                    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8},
                    {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8}
                ]
            }
        },
        'LightGBM': {
            'best_params': {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
            'best_score': 0.921,
            'cv_results': {
                'mean_test_score': [0.908, 0.921, 0.915, 0.918, 0.913, 0.916],
                'std_test_score': [0.007, 0.005, 0.006, 0.004, 0.007, 0.005],
                'params': [
                    {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
                    {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
                    {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8}
                ]
            }
        }
    }
    
    # Enhanced dataset info with more comprehensive details
    dataset_info = {
        'total_samples': 18846,
        'num_classes': 20,
        'class_distribution': {name: 942 for name in target_names},
        'avg_text_length': 221.3,
        'min_text_length': 1,
        'max_text_length': 1892,
        'class_names': target_names,
        'text_length_distribution': {
            'short': 4234,  # < 100 chars
            'medium': 8476,  # 100-300 chars
            'long': 6136     # > 300 chars
        },
        'vectorization_info': {
            'vocabulary_size': 4876,
            'max_features': 5000,
            'ngram_range': [1, 2],
            'min_df': 2,
            'max_df': 0.95,
            'avg_features_per_document': 156.7,
            'sparsity': 0.968,
            'top_features': [
                ['the', 12543], ['to', 9876], ['of', 8765], ['and', 7654], ['in', 6543],
                ['is', 5432], ['that', 4321], ['it', 3456], ['for', 2345], ['you', 1234],
                ['have', 1123], ['with', 1098], ['he', 987], ['as', 876], ['on', 765],
                ['be', 654], ['at', 543], ['this', 432], ['his', 321], ['from', 210]
            ],
            'feature_types': {
                'unigrams': 3876,
                'bigrams': 1000
            }
        },
        'data_quality': {
            'missing_values': 0,
            'duplicate_texts': 23,
            'empty_texts': 0,
            'avg_words_per_document': 45.2,
            'unique_words': 4876,
            'total_words': 852341
        }
    }
    
    # Enhanced data split info
    data_split_info = {
        'total_samples': 18846,
        'train_samples': 12061,
        'val_samples': 3015,
        'test_samples': 3770,
        'train_percentage': 64.0,
        'val_percentage': 16.0,
        'test_percentage': 20.0,
        'stratified_splitting': True,
        'class_distribution_train': {name: 603 for name in target_names},
        'class_distribution_val': {name: 151 for name in target_names},
        'class_distribution_test': {name: 188 for name in target_names},
        'split_ratios': {
            'train_val_test': [0.64, 0.16, 0.20],
            'train_test': [0.80, 0.20]
        }
    }
    
    # Complete output with enhanced metadata
    output = {
        'dataset_info': dataset_info,
        'model_results': model_results,
        'target_names': target_names,
        'hyperparameter_tuning': hyperparameter_tuning,
        'data_split_info': data_split_info,
        'metadata': {
            'computed_at': '2025-01-27T19:00:00Z',
            'version': '2.0.0',
            'description': 'Comprehensive ML results for 20 Newsgroups classification (static data) - All 10 Algorithms with Enhanced Data',
            'data_completeness': {
                'confusion_matrices': True,
                'roc_curves': True,
                'feature_importance': True,
                'hyperparameter_tuning': True,
                'performance_metrics': True,
                'dataset_statistics': True
            },
            'total_models': 10,
            'total_classes': 20,
            'total_samples': 18846
        }
    }
    
    # Save the complete results
    output_file = data_dir / 'ml_results.json'
    print(f"Saving comprehensive sample data to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("✅ Comprehensive sample data generated successfully!")
    print(f"📁 Data saved to: {data_dir}")
    print(f"📊 Models included: {len(model_results)}")
    print(f"🎯 Hyperparameter tuning results: {len(hyperparameter_tuning)}")
    print(f"📈 Best CV score: {max([hyperparameter_tuning[model]['best_score'] for model in hyperparameter_tuning])*100:.1f}%")
    print(f"📋 Confusion matrices: {len(model_results)}")
    print(f"📈 ROC curves: {len(model_results)}")
    print(f"🔍 Feature importance: {len(model_results)}")
    
    # Print performance summary
    print("\n🏆 Performance Summary (All 10 Algorithms):")
    sorted_models = sorted(hyperparameter_tuning.items(), key=lambda x: x[1]['best_score'], reverse=True)
    for i, (model_name, tuning_data) in enumerate(sorted_models, 1):
        best_score = tuning_data['best_score'] * 100
        print(f"  {i:2d}. {model_name:<20} {best_score:5.1f}%")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {dataset_info['total_samples']:,}")
    print(f"  Classes: {dataset_info['num_classes']}")
    print(f"  Vocabulary size: {dataset_info['vectorization_info']['vocabulary_size']:,}")
    print(f"  Average text length: {dataset_info['avg_text_length']:.1f} characters")

if __name__ == "__main__":
    generate_sample_data() 