'use client';

import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Button,
  Box,
  CircularProgress,
  Alert,
  Grid,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import { PlayArrow, Refresh, Cached, Clear } from '@mui/icons-material';
import ModelComparisonChart from '@/components/ModelComparisonChart';
import DatasetInfo from '@/components/DatasetInfo';
import MetricsTable from '@/components/MetricsTable';
import ConfusionMatrix from '@/components/ConfusionMatrix';
import ROCCurve from '@/components/ROCCurve';

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

interface CacheMetadata {
  cached: boolean;
  cache_available: boolean;
  ttl_seconds?: number;
  source: 'redis_cache' | 'fresh_analysis';
}

interface MLResults {
  dataset_info: DatasetInfo;
  model_results: Record<string, ModelResult>;
  target_names: string[];
  vectorization_info?: any;
  data_split_info?: DataSplitInfo;
  cache_metadata?: CacheMetadata;
}

export default function Home() {
  const [results, setResults] = useState<MLResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [cacheStatus, setCacheStatus] = useState<CacheMetadata | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/ml-results');
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || data.error || 'Failed to fetch data');
      }
      
      setResults(data);
      setCacheStatus(data.cache_metadata || null);
      
      // Set the first model as selected if none is selected
      if (!selectedModel && data.model_results) {
        const firstModel = Object.keys(data.model_results)[0];
        if (firstModel) {
          setSelectedModel(firstModel);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const clearCache = async () => {
    try {
      const response = await fetch('/api/clear-cache', { method: 'POST' });
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || 'Failed to clear cache');
      }
      
      // Refresh the data after clearing cache
      await fetchData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to clear cache');
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const formatTTL = (seconds?: number) => {
    if (!seconds || seconds <= 0) return 'Expired';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          20 Newsgroups ML Analysis
        </Typography>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          Text Classification with Multiple Algorithms
        </Typography>
        
        {/* Cache Status and Controls */}
        <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            startIcon={<PlayArrow />}
            onClick={fetchData}
            disabled={loading}
            sx={{ minWidth: 120 }}
          >
            {loading ? <CircularProgress size={20} /> : 'Run Analysis'}
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={fetchData}
            disabled={loading}
          >
            Refresh
          </Button>
          
          {cacheStatus && (
            <Chip
              icon={<Cached />}
              label={
                cacheStatus.cached 
                  ? `Cached (${formatTTL(cacheStatus.ttl_seconds)})`
                  : 'Fresh Analysis'
              }
              color={cacheStatus.cached ? 'success' : 'default'}
              variant="outlined"
            />
          )}
          
          {cacheStatus?.cache_available && (
            <Tooltip title="Clear cached results">
              <IconButton
                onClick={clearCache}
                color="warning"
                size="small"
              >
                <Clear />
              </IconButton>
            </Tooltip>
          )}
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress />
        </Box>
      )}

      {results && (
        <>
          {/* Dataset Information */}
          <Box sx={{ mb: 4 }}>
            <DatasetInfo
              datasetInfo={results.dataset_info}
              vectorizationInfo={results.vectorization_info}
              dataSplitInfo={results.data_split_info}
            />
          </Box>

          {/* Model Comparison Chart */}
          <Box sx={{ mb: 4 }}>
            <ModelComparisonChart modelResults={results.model_results} />
          </Box>

          {/* Metrics Table */}
          <Box sx={{ mb: 4 }}>
            <MetricsTable
              modelResults={results.model_results}
              targetNames={results.target_names}
            />
          </Box>

          {/* Confusion Matrix */}
          <Box sx={{ mb: 4 }}>
            <ConfusionMatrix
              modelResults={results.model_results}
              targetNames={results.target_names}
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
            />
          </Box>

          {/* ROC Curve */}
          <Box sx={{ mb: 4 }}>
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
