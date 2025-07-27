import sys
import json
import time
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env.local
load_dotenv('../.env.local')

# Try to import advanced algorithms (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available - skipping XGBoost algorithm", file=sys.stderr)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available - skipping LightGBM algorithm", file=sys.stderr)

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available - skipping CatBoost algorithm", file=sys.stderr)

def load_data():
    """Load the 20 newsgroups dataset"""
    print("Loading 20 newsgroups dataset...", file=sys.stderr)
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    return newsgroups

def preprocess_data(newsgroups):
    """Preprocess the text data with detailed vectorization"""
    print("Preprocessing data...", file=sys.stderr)
    
    # Split the data into training, validation, and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        newsgroups.data, 
        newsgroups.target, 
        test_size=0.2, 
        random_state=42,
        stratify=newsgroups.target
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2, # 0.2 of 0.8 = 0.16 of total
        random_state=42,
        stratify=y_train_full
    )
    
    # Vectorize the text data with detailed configuration
    vectorizer = TfidfVectorizer(
        max_features=5000, 
        stop_words='english', 
        ngram_range=(1, 2), 
        min_df=2, 
        max_df=0.95
    )
    
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_val_vectorized = vectorizer.transform(X_val)
    X_test_vectorized = vectorizer.transform(X_test)
    
    feature_names = vectorizer.get_feature_names_out()
    target_names = list(newsgroups.target_names)
    
    print(f"Vectorization complete - Vocabulary size: {len(vectorizer.vocabulary_)}", file=sys.stderr)
    print(f"Training set: {X_train_vectorized.shape}, Validation set: {X_val_vectorized.shape}, Test set: {X_test_vectorized.shape}", file=sys.stderr)
    
    return (X_train_vectorized, X_val_vectorized, X_test_vectorized, 
            y_train, y_val, y_test, vectorizer, feature_names, target_names)

def cache_results(results, target_names, dataset_info=None, data_split_info=None, hyperparameter_tuning=None):
    """Save results to a static file (Redis caching removed)"""
    try:
        # Prepare data for saving
        cache_data = {
            'model_results': results,
            'target_names': list(target_names),
            'timestamp': time.time(),
            'dataset_info': dataset_info,
            'data_split_info': data_split_info,
            'hyperparameter_tuning': hyperparameter_tuning
        }
        
        # Save to a static file instead of Redis
        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml_results.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print("  Results saved to static file", file=sys.stderr)
        
    except Exception as e:
        print(f"  Error saving results: {str(e)}", file=sys.stderr)

def train_models(X_train, X_val, X_test, y_train, y_val, y_test, target_names, dataset_info=None, data_split_info=None, hyperparameter_tuning=None):
    """Train multiple classification models with comprehensive evaluation"""
    print("Training models...", file=sys.stderr)
    
    # Binarize the labels for ROC curve calculation
    y_test_bin = label_binarize(y_test, classes=range(len(target_names)))
    n_classes = len(target_names)
    
    models = {
        # 1. Logistic Regression
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0, multi_class='ovr'),
        
        # 2. Support Vector Machine (using LinearSVC for speed)
        'Support Vector Machine': LinearSVC(random_state=42, max_iter=1000, C=1.0, loss='squared_hinge', dual=False, tol=1e-4),
        
        # 3. K-Nearest Neighbors
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', metric='minkowski', p=2),
        
        # 4. Naive Bayes
        'Naive Bayes': MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None),
        
        # 5. Decision Trees
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, criterion='gini', min_samples_split=2, min_samples_leaf=1, max_features=None),
        
        # 6. Random Forest
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8, criterion='gini', min_samples_split=5, min_samples_leaf=2, max_features='sqrt', n_jobs=-1),
        
        # 7. AdaBoost (optimized for speed)
        'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=42, min_samples_split=10), n_estimators=25, learning_rate=0.5, random_state=42, algorithm='SAMME'),
        
        # 8. Gradient Boosting (sklearn implementation - optimized)
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=15,  # Further reduced from 25
            random_state=42, 
            max_depth=2,  # Further reduced from 3
            learning_rate=0.3,  # Increased from 0.2 for faster convergence
            subsample=0.8,
            min_samples_split=20,  # Increased from 10
            min_samples_leaf=10,  # Increased from 5
            max_features='sqrt',  # Added for speed
            validation_fraction=0.1,  # Added for early stopping
            n_iter_no_change=5,  # Added for early stopping
            tol=1e-4  # Added for early stopping
        )
    }
    
    # Add advanced algorithms if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=15,  # Further reduced from 25
            max_depth=2,  # Further reduced from 3
            learning_rate=0.3,  # Increased from 0.2
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=-1,  # Use all CPU cores
            tree_method='hist'  # Faster tree method
        )
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=15,  # Further reduced from 25
            max_depth=2,  # Further reduced from 3
            learning_rate=0.3,  # Increased from 0.2
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='multiclass',
            num_class=len(target_names),
            n_jobs=-1,  # Use all CPU cores
            verbose=-1  # Suppress output
        )
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostClassifier(
            iterations=15,  # Further reduced from 25
            depth=2,  # Further reduced from 3
            learning_rate=0.3,  # Increased from 0.2
            random_state=42,
            subsample=0.8,
            colsample_bylevel=0.8,
            reg_lambda=1.0,
            loss_function='MultiClass',
            classes_count=len(target_names),
            thread_count=-1,  # Use all CPU cores
            verbose=False  # Suppress output
        )
    
    results = {}
    
    for name, model in models.items():
        try:
            print(f"  Training {name}...", file=sys.stderr)
            
            # Record training start time
            start_time = time.time()
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Record training time
            training_time = time.time() - start_time
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Calculate precision, recall, and F1-score
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # Generate ROC curves if possible
            roc_data = generate_roc_curves(model, X_test, y_test_bin, n_classes)
            
            # Store results
            results[name] = {
                'accuracy': float(test_accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                'validation_accuracy': float(val_accuracy),  # Add validation accuracy
                'training_time': float(training_time),  # Add training time
                'predictions': y_test_pred.tolist() if hasattr(y_test_pred, 'tolist') else list(y_test_pred),
                'true_labels': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                'roc_data': roc_data
            }
            
            # Cache the results after each model completes
            cache_results(results, target_names, dataset_info, data_split_info, hyperparameter_tuning)
            
        except Exception as e:
            print(f"  Error training {name}: {str(e)}", file=sys.stderr)
            continue
    
    return results

def generate_roc_curves(model, X_test, y_test_bin, n_classes):
    """Generate ROC curves for multiclass classification"""
    try:
        # Get probability predictions if available
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
        else:
            # For models without predict_proba, use decision_function
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
                # Reshape if needed for binary case
                if y_score.ndim == 1:
                    y_score = y_score.reshape(-1, 1)
            else:
                return None
        
        # Calculate ROC curves for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        return {
            'fpr': {str(k): v.tolist() if hasattr(v, 'tolist') else list(v) for k, v in fpr.items()},
            'tpr': {str(k): v.tolist() if hasattr(v, 'tolist') else list(v) for k, v in tpr.items()},
            'auc': {str(k): float(v) for k, v in roc_auc.items()}
        }
    except Exception as e:
        print(f"  Error generating ROC curves: {str(e)}", file=sys.stderr)
        return None

def hyperparameter_tuning(X_train, X_val, y_train, y_val, target_names):
    """Perform hyperparameter tuning on the best performing models"""
    print("Performing hyperparameter tuning...", file=sys.stderr)
    
    # Combine training and validation data for cross-validation
    X_combined = np.vstack([X_train.toarray(), X_val.toarray()])
    y_combined = np.concatenate([y_train, y_val])
    
    tuning_results = {}
    
    # 1. Logistic Regression Tuning (reduced grid)
    print("  Tuning Logistic Regression...", file=sys.stderr)
    lr_param_grid = {
        'C': [0.1, 1.0, 5.0],
        'penalty': ['l2'],
        'solver': ['liblinear'],
        'max_iter': [1000]
    }
    
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42, multi_class='ovr'),
        lr_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=2,  # Reduced from -1 to 2
        verbose=0
    )
    
    lr_grid.fit(X_combined, y_combined)
    tuning_results['Logistic Regression'] = {
        'best_params': lr_grid.best_params_,
        'best_score': float(lr_grid.best_score_),
        'cv_results': {
            'mean_test_score': lr_grid.cv_results_['mean_test_score'].tolist(),
            'std_test_score': lr_grid.cv_results_['std_test_score'].tolist(),
            'params': lr_grid.cv_results_['params']
        }
    }
    
    # 2. Random Forest Tuning (reduced grid)
    print("  Tuning Random Forest...", file=sys.stderr)
    rf_param_grid = {
        'n_estimators': [25, 50],
        'max_depth': [5, 8, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=2),  # Reduced from -1 to 2
        rf_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=2,  # Reduced from -1 to 2
        verbose=0
    )
    
    rf_grid.fit(X_combined, y_combined)
    tuning_results['Random Forest'] = {
        'best_params': rf_grid.best_params_,
        'best_score': float(rf_grid.best_score_),
        'cv_results': {
            'mean_test_score': rf_grid.cv_results_['mean_test_score'].tolist(),
            'std_test_score': rf_grid.cv_results_['std_test_score'].tolist(),
            'params': rf_grid.cv_results_['params']
        }
    }
    
    # 3. Support Vector Machine Tuning (reduced grid)
    print("  Tuning Support Vector Machine...", file=sys.stderr)
    svm_param_grid = {
        'C': [0.5, 1.0, 2.0],
        'loss': ['squared_hinge'],
        'max_iter': [1000]
    }
    
    svm_grid = GridSearchCV(
        LinearSVC(random_state=42, dual=False, tol=1e-4),
        svm_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=2,  # Reduced from -1 to 2
        verbose=0
    )
    
    svm_grid.fit(X_combined, y_combined)
    tuning_results['Support Vector Machine'] = {
        'best_params': svm_grid.best_params_,
        'best_score': float(svm_grid.best_score_),
        'cv_results': {
            'mean_test_score': svm_grid.cv_results_['mean_test_score'].tolist(),
            'std_test_score': svm_grid.cv_results_['std_test_score'].tolist(),
            'params': svm_grid.cv_results_['params']
        }
    }
    
    # 4. Gradient Boosting Tuning (if available) - reduced grid
    if XGBOOST_AVAILABLE:
        print("  Tuning XGBoost...", file=sys.stderr)
        xgb_param_grid = {
            'n_estimators': [25, 50],
            'max_depth': [2, 3],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False, n_jobs=2),  # Reduced from -1 to 2
            xgb_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=2,  # Reduced from -1 to 2
            verbose=0
        )
        
        xgb_grid.fit(X_combined, y_combined)
        tuning_results['XGBoost'] = {
            'best_params': xgb_grid.best_params_,
            'best_score': float(xgb_grid.best_score_),
            'cv_results': {
                'mean_test_score': xgb_grid.cv_results_['mean_test_score'].tolist(),
                'std_test_score': xgb_grid.cv_results_['std_test_score'].tolist(),
                'params': xgb_grid.cv_results_['params']
            }
        }
    
    # 5. K-Nearest Neighbors Tuning (reduced grid)
    print("  Tuning K-Nearest Neighbors...", file=sys.stderr)
    knn_param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski']
    }
    
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=2,  # Reduced from -1 to 2
        verbose=0
    )
    
    knn_grid.fit(X_combined, y_combined)
    tuning_results['K-Nearest Neighbors'] = {
        'best_params': knn_grid.best_params_,
        'best_score': float(knn_grid.best_score_),
        'cv_results': {
            'mean_test_score': knn_grid.cv_results_['mean_test_score'].tolist(),
            'std_test_score': knn_grid.cv_results_['std_test_score'].tolist(),
            'params': knn_grid.cv_results_['params']
        }
    }
    
    print("Hyperparameter tuning completed!", file=sys.stderr)
    return tuning_results

def get_dataset_info(newsgroups, vectorizer, feature_names):
    """Get detailed information about the dataset and vectorization"""
    print("Getting dataset information...", file=sys.stderr)
    
    # Class distribution
    class_counts = {}
    for i, name in enumerate(newsgroups.target_names):
        count = (newsgroups.target == i).sum()
        class_counts[name] = int(count)
    
    # Text length statistics
    text_lengths = [len(text.split()) for text in newsgroups.data]
    
    # Vectorization statistics
    vocab_size = len(vectorizer.vocabulary_)
    avg_features_per_doc = vectorizer.transform(newsgroups.data).mean(axis=0).A1.mean()
    
    # Most common features
    feature_frequencies = vectorizer.transform(newsgroups.data).sum(axis=0).A1
    top_features_idx = np.argsort(feature_frequencies)[-10:][::-1]
    top_features = [feature_names[i] for i in top_features_idx]
    top_frequencies = [int(feature_frequencies[i]) for i in top_features_idx]
    
    return {
        'total_samples': len(newsgroups.data),
        'num_classes': len(newsgroups.target_names),
        'class_distribution': class_counts,
        'avg_text_length': float(np.mean(text_lengths)),
        'min_text_length': int(np.min(text_lengths)),
        'max_text_length': int(np.max(text_lengths)),
        'class_names': list(newsgroups.target_names),
        'vectorization_info': {
            'vocabulary_size': vocab_size,
            'max_features': 5000,
            'ngram_range': [1, 2],
            'min_df': 2,
            'max_df': 0.95,
            'avg_features_per_document': float(avg_features_per_doc),
            'top_features': list(zip(top_features, top_frequencies))
        }
    }

def main():
    """Main function to run the machine learning pipeline"""
    try:
        # Load data
        newsgroups = load_data()
        
        # Preprocess data with detailed vectorization
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         vectorizer, feature_names, target_names) = preprocess_data(newsgroups)
        
        # Get detailed dataset info including vectorization
        dataset_info = get_dataset_info(newsgroups, vectorizer, feature_names)
        
        # Perform hyperparameter tuning
        tuning_results = hyperparameter_tuning(X_train, X_val, y_train, y_val, target_names)
        
        # Train models with validation
        results = train_models(X_train, X_val, X_test, y_train, y_val, y_test, target_names, dataset_info, {
            'total_samples': len(newsgroups.data),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'train_percentage': round(len(y_train) / len(newsgroups.data) * 100, 1),
            'val_percentage': round(len(y_val) / len(newsgroups.data) * 100, 1),
            'test_percentage': round(len(y_test) / len(newsgroups.data) * 100, 1)
        }, tuning_results)
        
        # Prepare output
        output = {
            'dataset_info': dataset_info,
            'model_results': results,
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
            }
        }
        
        # Print results to stdout for Next.js to capture
        print(json.dumps(output))
        
    except Exception as e:
        error_output = {
            'error': str(e),
            'type': 'error'
        }
        print(json.dumps(error_output))
        sys.exit(1)

if __name__ == "__main__":
    main() 