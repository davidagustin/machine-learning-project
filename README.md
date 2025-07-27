# Machine Learning Project - 20 Newsgroups Classification

A comprehensive machine learning analysis dashboard for the 20 Newsgroups dataset, featuring hyperparameter tuning, model comparison, and interactive visualizations.

## 🚀 Features

### 📊 **Dataset Information**
- Complete dataset statistics and analysis
- Text vectorization details (TF-IDF)
- Class distribution visualization
- Data split information (train/validation/test)

### 🎯 **Hyperparameter Tuning**
- **Grid Search CV** optimization for multiple algorithms
- **Best parameters** display with cross-validation scores
- **Detailed results** for each model's tuning process
- **Interactive cards** showing optimization progress

### 📈 **Model Comparison**
- **10 algorithms** including Logistic Regression, Random Forest, SVM, XGBoost, KNN, Naive Bayes, Decision Tree, AdaBoost, Gradient Boosting, and LightGBM
- **Performance metrics** (accuracy, precision, recall, F1-score)
- **Cross-validation** results with confidence intervals
- **Interactive charts** and visualizations

### 🎨 **Visualizations**
- **Confusion matrices** with improved label readability
- **ROC curves** for multiclass classification
- **Model comparison charts** with responsive design
- **Metrics tables** with sorting and filtering

## 🛠️ Technical Stack

- **Frontend**: Next.js 14, React, Material-UI
- **Charts**: Chart.js with react-chartjs-2
- **Data**: Static JSON files (pre-computed results)
- **Styling**: Responsive design with mobile optimization

## 📁 Project Structure

```
machine-learning-project/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Main dashboard page
│   └── layout.tsx         # App layout
├── components/            # React components
│   ├── DatasetInfo.tsx    # Dataset information card
│   ├── HyperparameterTuning.tsx  # Hyperparameter tuning results
│   ├── ModelComparisonChart.tsx  # Model comparison visualization
│   ├── MetricsTable.tsx   # Performance metrics table
│   ├── ConfusionMatrix.tsx # Confusion matrix visualization
│   └── ROCCurve.tsx       # ROC curve visualization
├── data/                  # Static data files
│   └── ml_results.json    # Pre-computed ML results
├── scripts/               # Data generation scripts
│   ├── generate_sample_data.py  # Sample data generator
│   └── ml_processor.py    # Full ML pipeline (optional)
└── pages/api/             # API endpoints
    ├── ml-results.ts      # Serves static ML data
    └── clear-cache.ts     # Regenerates sample data
```

## 🚀 **Quick Start**

1. **Clone and Install:**
   ```bash
   git clone <repository-url>
   cd machine-learning-project
   npm install
   ```

2. **Generate Sample Data:**
   ```bash
   cd scripts
   python3 generate_sample_data.py
   cd ..
   ```

3. **Run the Application:**
   ```bash
   npm run dev
   ```

4. **Access the Dashboard:**
   - Open [http://localhost:3000](http://localhost:3000)
   - View comprehensive ML results for all 10 algorithms
   - Explore hyperparameter tuning results
   - Analyze performance metrics and visualizations

## 📊 **Enhanced Data Features**

### **🎯 Comprehensive Model Analysis**
- **10 algorithms** with full performance metrics
- **Confusion matrices** for all models (20x20)
- **ROC curves** with AUC scores for each class
- **Feature importance** rankings (top 50 features)
- **Training/prediction times** and model sizes
- **Cross-validation** results with confidence intervals

### **📈 Advanced Visualizations**
- **Interactive confusion matrices** with smart label truncation
- **Multi-class ROC curves** with individual class performance
- **Model comparison charts** with error bars
- **Performance metrics tables** with detailed statistics
- **Hyperparameter tuning results** with parameter rankings

### **📋 Rich Dataset Information**
- **Text length distribution** (short/medium/long)
- **Vocabulary statistics** (4,876 unique words)
- **Feature engineering details** (unigrams/bigrams)
- **Data quality metrics** (missing values, duplicates)
- **Stratified split information** with class distributions

## 📊 Hyperparameter Tuning Results

The application includes comprehensive hyperparameter tuning for all 10 algorithms:

### **🏆 Performance Rankings:**

1. **LightGBM** (92.1% CV Score)
   - **Best Parameters**: n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8

2. **XGBoost** (91.5% CV Score)
   - **Best Parameters**: n_estimators=50, max_depth=3, learning_rate=0.2, subsample=0.8, colsample_bytree=0.8

3. **Gradient Boosting** (90.3% CV Score)
   - **Best Parameters**: n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8

4. **Random Forest** (89.2% CV Score)
   - **Best Parameters**: n_estimators=50, max_depth=8, min_samples_split=5, min_samples_leaf=2, max_features='sqrt'

5. **Logistic Regression** (84.7% CV Score)
   - **Best Parameters**: C=1.0, penalty='l2', solver='liblinear', max_iter=1000

6. **Support Vector Machine** (82.3% CV Score)
   - **Best Parameters**: C=1.0, loss='squared_hinge', max_iter=1000

7. **AdaBoost** (79.8% CV Score)
   - **Best Parameters**: n_estimators=50, learning_rate=0.5, algorithm='SAMME'

8. **K-Nearest Neighbors** (75.6% CV Score)
   - **Best Parameters**: n_neighbors=5, weights='uniform', metric='minkowski'

9. **Naive Bayes** (73.4% CV Score)
   - **Best Parameters**: alpha=0.5, fit_prior=True

10. **Decision Tree** (68.7% CV Score)
    - **Best Parameters**: max_depth=10, min_samples_split=5, min_samples_leaf=2, criterion='gini'

## 🎨 UI Features

### **Responsive Design**
- **Mobile-optimized** layouts and components
- **Adaptive charts** that resize for different screen sizes
- **Touch-friendly** interactions

### **Interactive Elements**
- **Expandable sections** for detailed hyperparameter results
- **Sortable tables** for model comparison
- **Model selection** for confusion matrices and ROC curves
- **Progress indicators** for tuning results

### **Visual Improvements**
- **Enhanced label readability** on confusion matrices
- **Smart truncation** for long class names
- **Color-coded** performance indicators
- **Professional styling** with Material-UI components

## 🔧 Configuration

### **Static Data Approach**
The application uses pre-computed static data for fast loading:

- **No server-side computation** during requests
- **Instant response times** for all visualizations
- **Consistent data** across all sessions
- **Easy to update** by regenerating sample data

### **Data Regeneration**
To update the sample data:
```bash
# Via API
curl -X POST http://localhost:3000/api/clear-cache

# Or manually
cd scripts && python3 generate_sample_data.py
```

## 📈 Performance

- **Fast loading**: Static data eliminates computation delays
- **Responsive UI**: Optimized for all device sizes
- **Smooth interactions**: Real-time chart updates
- **Efficient rendering**: Optimized React components

## 🎯 Key Improvements

### **Hyperparameter Tuning Card**
- **Comprehensive results** display
- **Best parameters** with cross-validation scores
- **Expandable details** for each model
- **Visual progress indicators**

### **Enhanced Readability**
- **Improved X-axis labels** on confusion matrices
- **Smart truncation** for long class names
- **Better spacing** and typography
- **Mobile-friendly** text sizing

### **Static Data Architecture**
- **Pre-computed results** for instant loading
- **No computational bottlenecks**
- **Consistent performance**
- **Easy maintenance**

## 🔮 Future Enhancements

- **Real-time model training** (optional)
- **Additional algorithms** (LightGBM, CatBoost)
- **Advanced visualizations** (SHAP plots, feature importance)
- **Export functionality** for results
- **Custom dataset upload**

## 📝 License

This project is open source and available under the MIT License.
