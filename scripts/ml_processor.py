import sys
import json
import time
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, cross_val_score
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

def cache_results(results, target_names):
    """Cache the current results to Redis after each model completes"""
    try:
        import redis
        import os
        
        # Get Redis URL from environment variable
        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            print("  No REDIS_URL found, skipping caching", file=sys.stderr)
            return
        
        # Connect to Redis
        r = redis.from_url(redis_url)
        
        # Prepare data for caching
        cache_data = {
            'model_results': results,
            'target_names': list(target_names),
            'timestamp': time.time(),
            'models_completed': len(results)
        }
        
        # Cache the data
        r.setex('ml_analysis_results', 3600, json.dumps(cache_data))  # Cache for 1 hour
        print(f"  Cached results for {len(results)} models", file=sys.stderr)
        
    except Exception as e:
        print(f"  Error caching results: {str(e)}", file=sys.stderr)

def train_models(X_train, X_val, X_test, y_train, y_val, y_test, target_names):
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
            num_class=n_classes,
            n_jobs=-1,  # Use all CPU cores
            min_child_samples=20  # Increased from 10
        )
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostClassifier(
            iterations=15,  # Further reduced from 25
            depth=2,  # Further reduced from 3
            learning_rate=0.3,  # Increased from 0.2
            random_state=42,
            colsample_bylevel=0.8,
            reg_lambda=1.0,
            verbose=False,
            thread_count=-1,  # Use all CPU cores
            grow_policy='Lossguide',  # Faster growing policy
            min_data_in_leaf=20,  # Increased from 10
            bootstrap_type='Bernoulli',  # Explicit bootstrap type
            subsample=0.8  # Now compatible with Bernoulli bootstrap
        )
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...", file=sys.stderr)
        
        try:
            # Start timing
            start_time = time.time()
            
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            y_test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            roc_data = generate_roc_curves(model, X_test, y_test_bin, n_classes)
            
            # End timing
            end_time = time.time()
            training_time = end_time - start_time
            
            print(f"  {name} - Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, Time: {training_time:.2f}s", file=sys.stderr)
            
            results[name] = {
                'accuracy': float(test_accuracy),
                'validation_accuracy': float(val_accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'training_time': float(training_time),  # Add training time
                'predictions': y_test_pred.tolist() if hasattr(y_test_pred, 'tolist') else list(y_test_pred),
                'true_labels': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                'roc_data': roc_data
            }
            
            # Cache the results after each model completes
            cache_results(results, target_names)
            
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
        
        # Train models with validation
        results = train_models(X_train, X_val, X_test, y_train, y_val, y_test, target_names)
        
        # Prepare output
        output = {
            'dataset_info': dataset_info,
            'model_results': results,
            'target_names': list(target_names),
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