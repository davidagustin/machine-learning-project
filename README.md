# 20 Newsgroups Machine Learning Analysis

A comprehensive Next.js application that performs text classification on the 20 Newsgroups dataset using multiple machine learning algorithms. The application features a modern React frontend with Material-UI components and interactive visualizations.

## 🚀 Features

- **Multiple ML Algorithms**: Naive Bayes, Logistic Regression, Random Forest
- **Interactive Visualizations**: Model comparison charts, confusion matrices, ROC curves
- **Real-time Analysis**: Fast processing with Redis caching
- **Responsive Design**: Works on desktop and mobile devices
- **Professional UI**: Material-UI components with modern styling

## 📊 Dataset

The application uses the **20 Newsgroups dataset** from scikit-learn, which contains:
- **18,846 documents** across 20 different newsgroups
- **Text classification** tasks with balanced class distribution
- **Preprocessed text** with TF-IDF vectorization

## 🛠️ Installation

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+ with pip
- Redis (local or cloud service like Vercel Redis)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd machine-learning-project
```

### 2. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Configuration

#### Option A: Using the Setup Script (Recommended)

**Unix/macOS/Linux:**
```bash
# Run the setup script to create .env.local
./setup-env.sh

# Edit .env.local with your actual Redis URL
nano .env.local  # or use your preferred editor
```

**Windows:**
```cmd
# Run the setup script to create .env.local
setup-env.bat

# Edit .env.local with your actual Redis URL
notepad .env.local  # or use your preferred editor
```

#### Option B: Manual Setup

Create a `.env.local` file in the root directory with your Redis configuration:

```bash
# .env.local
REDIS_URL="redis://default:your_password@your_host:your_port"
```

**Important Security Notes:**
- The `.env.local` file is automatically ignored by git (see `.gitignore`)
- Never commit your actual Redis credentials to version control
- For production, use environment variables in your deployment platform

### 4. Redis Setup

#### Option A: Local Redis (Development)
```bash
# Install Redis (macOS)
brew install redis
brew services start redis

# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server
sudo systemctl start redis-server
```

#### Option B: Vercel Redis (Production)
1. Create a Vercel Redis database in your Vercel dashboard
2. Copy the connection string to your `.env.local` file
3. The connection string format is: `redis://default:password@host:port`

### 5. Run the Application

```bash
# Start the development server
npm run dev

# Open http://localhost:3000 in your browser
```

## 📁 Project Structure

```
machine-learning-project/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Main dashboard page
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── DatasetInfo.tsx    # Dataset statistics
│   ├── ModelComparisonChart.tsx
│   ├── MetricsTable.tsx   # Performance metrics
│   ├── ConfusionMatrix.tsx
│   └── ROCCurve.tsx
├── pages/api/            # API routes
│   ├── ml-results.ts     # ML analysis endpoint
│   └── clear-cache.ts    # Cache management
├── lib/                  # Utility libraries
│   └── redis.ts          # Redis configuration
├── scripts/              # Python ML scripts
│   └── ml_processor.py   # Core ML pipeline
├── requirements.txt      # Python dependencies
├── package.json          # Node.js dependencies
├── setup-env.sh         # Environment setup script (Unix/macOS/Linux)
├── setup-env.bat        # Environment setup script (Windows)
└── .env.local           # Environment variables (not in git)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `REDIS_URL` | Redis connection string | Yes |
| `REDIS_HOST` | Redis host (fallback) | No |
| `REDIS_PORT` | Redis port (fallback) | No |
| `REDIS_PASSWORD` | Redis password (fallback) | No |

### Cache Configuration

- **TTL**: 1 hour (3600 seconds)
- **Cache Key**: `ml_analysis_results`
- **Automatic**: Results cached after first analysis
- **Manual**: Clear cache via API or UI button

## 🚀 Deployment

### Vercel (Recommended)

1. **Connect Repository**: Link your GitHub repository to Vercel
2. **Environment Variables**: Add `REDIS_URL` in Vercel dashboard
3. **Deploy**: Vercel will automatically build and deploy

### Other Platforms

1. **Build**: `npm run build`
2. **Start**: `npm start`
3. **Environment**: Set `REDIS_URL` in your deployment platform

## 📊 API Endpoints

### GET `/api/ml-results`
Returns machine learning analysis results with caching.

**Response:**
```json
{
  "dataset_info": { ... },
  "model_results": { ... },
  "cache_metadata": {
    "cached": true,
    "cache_available": true,
    "ttl_seconds": 3600,
    "source": "redis_cache"
  }
}
```

### POST `/api/clear-cache`
Clears the ML results cache.

**Response:**
```json
{
  "message": "Cache cleared successfully",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "cache_available": true
}
```

## 🔍 Performance Metrics

The application tracks and displays:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score (weighted average)
- **Recall**: Recall score (weighted average)
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Validation**: 5-fold CV mean and standard deviation
- **Validation Accuracy**: Performance on validation set

## 📈 Visualization Features

- **Model Comparison**: Bar charts comparing algorithm performance
- **Confusion Matrix**: Heatmap visualization with diagonal emphasis
- **ROC Curves**: Multi-class ROC analysis
- **Dataset Statistics**: Comprehensive dataset information
- **Cache Status**: Real-time cache information and TTL

## 🔒 Security

- **Environment Variables**: Sensitive data stored in `.env.local` (gitignored)
- **Redis Security**: Use strong passwords and SSL connections in production
- **API Protection**: Consider adding authentication for production use
- **Input Validation**: All inputs are validated and sanitized

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Commit: `git commit -m 'Add feature'`
6. Push: `git push origin feature-name`
7. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check if Redis is running: `redis-cli ping`
   - Verify `REDIS_URL` in `.env.local`
   - Ensure Redis server is accessible

2. **Python Dependencies**
   - Activate virtual environment: `source venv/bin/activate`
   - Reinstall: `pip install -r requirements.txt`

3. **Cache Not Working**
   - Check Redis connection
   - Verify cache configuration in `lib/redis.ts`
   - Clear cache manually via API

4. **Build Errors**
   - Clear Next.js cache: `rm -rf .next`
   - Reinstall dependencies: `npm install`

5. **Environment Variables**
   - Run `./setup-env.sh` to create `.env.local`
   - Ensure `.env.local` contains your actual Redis URL
   - Check that `.env.local` is not being tracked by git

### Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open an issue on GitHub
