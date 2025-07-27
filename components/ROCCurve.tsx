'use client';

import { useState, useEffect } from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Paper,
  useTheme,
  useMediaQuery
} from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ModelResult {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  cv_mean: number;
  cv_std: number;
  predictions: number[];
  true_labels: number[];
  roc_data?: {
    fpr: { [key: string]: number[] };
    tpr: { [key: string]: number[] };
    auc: { [key: string]: number };
  };
}

interface ROCCurveProps {
  modelResults?: { [key: string]: ModelResult };
  targetNames?: string[];
  selectedModel?: string;
  onModelChange: (model: string) => void;
}

export default function ROCCurve({ 
  modelResults, 
  targetNames, 
  selectedModel, 
  onModelChange 
}: ROCCurveProps) {
  const [rocData, setRocData] = useState<any>(null);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));

  useEffect(() => {
    if (selectedModel && modelResults && modelResults[selectedModel] && modelResults[selectedModel].roc_data) {
      const result = modelResults[selectedModel];
      setRocData(result.roc_data);
    } else {
      setRocData(null);
    }
  }, [selectedModel, modelResults]);

  // Add null checks
  if (!modelResults || !targetNames || !selectedModel) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography color="text.secondary">
          Loading ROC curve data...
        </Typography>
      </Box>
    );
  }

  const modelNames = Object.keys(modelResults).filter(name => 
    modelResults[name].roc_data !== undefined
  );

  const createROCData = () => {
    if (!rocData) return { labels: [], datasets: [] };

    const datasets = Object.keys(rocData.fpr).map((classIndex, i) => {
      const classNum = parseInt(classIndex);
      const className = targetNames[classNum] || `Class ${classNum}`;
      const auc = rocData.auc[classIndex];
      
      return {
        label: `${className} (AUC: ${auc.toFixed(3)})`,
        data: rocData.fpr[classIndex].map((fpr: number, j: number) => ({
          x: fpr,
          y: rocData.tpr[classIndex][j]
        })),
        borderColor: `hsl(${(i * 360) / Object.keys(rocData.fpr).length}, 70%, 50%)`,
        backgroundColor: `hsla(${(i * 360) / Object.keys(rocData.fpr).length}, 70%, 50%, 0.1)`,
        tension: 0.1,
        pointRadius: 0
      };
    });

    // Add diagonal line for reference
    datasets.push({
      label: 'Random Classifier (AUC: 0.500)',
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: 'rgba(0, 0, 0, 0.3)',
      backgroundColor: 'rgba(0, 0, 0, 0.1)',
      pointRadius: 0
    } as any);

    return {
      labels: [],
      datasets
    };
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          font: {
            size: isSmallScreen ? 10 : isMobile ? 11 : 12
          }
        }
      },
      title: {
        display: true,
        text: `ROC Curves - ${selectedModel}`,
        font: {
          size: isSmallScreen ? 14 : isMobile ? 15 : 16
        }
      },
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        min: 0,
        max: 1,
        title: {
          display: true,
          text: 'False Positive Rate',
          font: {
            size: isSmallScreen ? 12 : isMobile ? 13 : 14
          }
        },
        ticks: {
          font: {
            size: isSmallScreen ? 10 : isMobile ? 11 : 12
          }
        }
      },
      y: {
        type: 'linear',
        min: 0,
        max: 1,
        title: {
          display: true,
          text: 'True Positive Rate',
          font: {
            size: isSmallScreen ? 12 : isMobile ? 13 : 14
          }
        },
        ticks: {
          font: {
            size: isSmallScreen ? 10 : isMobile ? 11 : 12
          }
        }
      }
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ 
        fontSize: isSmallScreen ? '1.25rem' : isMobile ? '1.5rem' : '1.75rem',
        mb: isMobile ? 1 : 2
      }}>
        ROC Curves
      </Typography>
      
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Select Model</InputLabel>
        <Select
          value={selectedModel}
          label="Select Model"
          onChange={(e) => onModelChange(e.target.value)}
        >
          {modelNames.map((name) => (
            <MenuItem key={name} value={name}>
              {name}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {rocData ? (
        <Paper elevation={2} sx={{ p: isMobile ? 1 : 2 }}>
          <Box sx={{ 
            height: isSmallScreen ? 400 : isMobile ? 500 : 600,
            width: '100%',
            minHeight: 300
          }}>
            <Line data={createROCData()} options={options} />
          </Box>
        </Paper>
      ) : (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography color="text.secondary">
            No ROC curve data available for the selected model.
          </Typography>
        </Box>
      )}
    </Box>
  );
} 