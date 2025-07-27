'use client';

import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  useTheme,
  useMediaQuery,
  IconButton,
  Tooltip
} from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';

import ModelComparisonChart from '@/components/ModelComparisonChart';
import DatasetInfo from '@/components/DatasetInfo';
import MetricsTable from '@/components/MetricsTable';
import ConfusionMatrix from '@/components/ConfusionMatrix';
import ROCCurve from '@/components/ROCCurve';
import HyperparameterTuning from '@/components/HyperparameterTuning';

interface ModelResult {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  cv_mean: number;
  cv_std: number;
  validation_accuracy?: number;
  predictions: number[];
  true_labels: number[];
  confusion_matrix?: number[][];
  roc_data?: {
    [key: string]: {
      fpr: number[];
      tpr: number[];
      auc: number;
    };
  };
  feature_importance?: Array<{
    feature: string;
    importance: number;
  }>;
  training_time?: number;
  prediction_time?: number;
  model_size_mb?: number;
}

interface DatasetInfo {
  total_samples: number;
  num_classes: number;
  class_distribution: Record<string, number>;
  avg_text_length: number;
  min_text_length: number;
  max_text_length: number;
  class_names: string[];
  vectorization_info?: {
    vocabulary_size: number;
    max_features: number;
    ngram_range: number[];
    min_df: number;
    max_df: number;
    avg_features_per_document: number;
    top_features: [string, number][];
  };
}

interface DataSplitInfo {
  total_samples: number;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  train_percentage: number;
  val_percentage: number;
  test_percentage: number;
}



interface MLResults {
  dataset_info: DatasetInfo;
  model_results: Record<string, ModelResult>;
  target_names: string[];
  vectorization_info?: any;
  data_split_info?: DataSplitInfo;

  hyperparameter_tuning?: any;
}

export default function Home() {
  const [results, setResults] = useState<MLResults | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/api/ml-results');
        const data = await response.json();
        
        if (!response.ok) {
          console.error('Failed to fetch data:', data);
          return;
        }
        
        setResults(data);
        
        // Set the first model as selected if none is selected
        if (!selectedModel && data.model_results) {
          const firstModel = Object.keys(data.model_results)[0];
          if (firstModel) {
            setSelectedModel(firstModel);
          }
        }
      } catch (err) {
        console.error('Error fetching data:', err);
      }
    };

    fetchData();
  }, [selectedModel]); // Include selectedModel in dependencies



  return (
    <Container maxWidth="xl" sx={{ 
      py: isSmallScreen ? 1 : isMobile ? 2 : 4, 
      px: isSmallScreen ? 0.5 : isMobile ? 1 : 2 
    }}>
      <Box sx={{ mb: isSmallScreen ? 1 : isMobile ? 2 : 4 }}>
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between', 
          mb: isSmallScreen ? 1 : 2,
          flexDirection: isSmallScreen ? 'column' : 'row',
          gap: isSmallScreen ? 1 : 0
        }}>
          <Box sx={{ textAlign: isSmallScreen ? 'center' : 'left', width: '100%' }}>
            <Typography 
              variant="h3" 
              component="h1" 
              gutterBottom
              sx={{ 
                fontSize: isSmallScreen ? '1.5rem' : isMobile ? '2rem' : '3rem',
                lineHeight: isSmallScreen ? 1.2 : 1.167,
                wordBreak: 'break-word'
              }}
            >
              20 Newsgroups ML Analysis
            </Typography>
            <Typography 
              variant="h6" 
              color="text.secondary" 
              gutterBottom
              sx={{ 
                fontSize: isSmallScreen ? '0.875rem' : isMobile ? '1rem' : '1.25rem',
                wordBreak: 'break-word'
              }}
            >
              Text Classification with Multiple Algorithms
            </Typography>
          </Box>
          
          <Tooltip title="View on GitHub">
            <IconButton
              href="https://github.com/davidagustin/machine-learning-project"
              target="_blank"
              rel="noopener noreferrer"
              sx={{
                color: 'text.secondary',
                '&:hover': {
                  color: 'primary.main',
                  transform: 'scale(1.1)',
                },
                transition: 'all 0.2s ease-in-out',
                minWidth: isSmallScreen ? '44px' : 'auto',
                minHeight: isSmallScreen ? '44px' : 'auto',
              }}
            >
              <GitHubIcon sx={{ fontSize: isSmallScreen ? '1.5rem' : '2rem' }} />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>



      {results && (
        <>
          {/* Dataset Information */}
          <Box sx={{ mb: isSmallScreen ? 1 : isMobile ? 2 : 4 }}>
            <DatasetInfo
              datasetInfo={results.dataset_info}
              vectorizationInfo={results.vectorization_info}
              dataSplitInfo={results.data_split_info}
            />
          </Box>

          {/* Hyperparameter Tuning Results */}
          <Box sx={{ mb: isSmallScreen ? 1 : isMobile ? 2 : 4 }}>
            <HyperparameterTuning
              hyperparameterTuning={results.hyperparameter_tuning}
            />
          </Box>

          {/* Model Comparison Chart */}
          <Box sx={{ mb: isSmallScreen ? 1 : isMobile ? 2 : 4 }}>
            <ModelComparisonChart modelResults={results.model_results} />
          </Box>

          {/* Metrics Table */}
          <Box sx={{ mb: isSmallScreen ? 1 : isMobile ? 2 : 4 }}>
            <MetricsTable
              modelResults={results.model_results}
              targetNames={results.target_names}
            />
          </Box>

          {/* Confusion Matrix */}
          <Box sx={{ mb: isSmallScreen ? 1 : isMobile ? 2 : 4 }}>
            <ConfusionMatrix
              modelResults={results.model_results}
              targetNames={results.target_names}
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
            />
          </Box>

          {/* ROC Curve */}
          <Box sx={{ mb: isSmallScreen ? 1 : isMobile ? 2 : 4 }}>
            <ROCCurve
              modelResults={results.model_results}
              targetNames={results.target_names}
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
            />
          </Box>
        </>
      )}
    </Container>
  );
}
