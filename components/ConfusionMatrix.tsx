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
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
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
import { Scatter } from 'react-chartjs-2';

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
}

interface ConfusionMatrixProps {
  modelResults?: { [key: string]: ModelResult };
  targetNames?: string[];
  selectedModel?: string;
  onModelChange: (model: string) => void;
}

export default function ConfusionMatrix({ 
  modelResults, 
  targetNames, 
  selectedModel, 
  onModelChange 
}: ConfusionMatrixProps) {
  const [confusionMatrix, setConfusionMatrix] = useState<number[][]>([]);

  useEffect(() => {
    if (selectedModel && modelResults && modelResults[selectedModel] && targetNames) {
      const result = modelResults[selectedModel];
      const matrix = computeConfusionMatrix(result.true_labels, result.predictions, targetNames.length);
      setConfusionMatrix(matrix);
    }
  }, [selectedModel, modelResults, targetNames]);

  // Add null checks
  if (!modelResults || !targetNames || !selectedModel) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography color="text.secondary">
          Loading confusion matrix data...
        </Typography>
      </Box>
    );
  }

  const computeConfusionMatrix = (trueLabels: number[], predictions: number[], numClasses: number) => {
    const matrix = Array(numClasses).fill(0).map(() => Array(numClasses).fill(0));
    
    for (let i = 0; i < trueLabels.length; i++) {
      matrix[trueLabels[i]][predictions[i]]++;
    }
    
    return matrix;
  };

  const modelNames = Object.keys(modelResults);

  // Create a proper heatmap visualization using scatter plot
  const createHeatmapData = () => {
    if (confusionMatrix.length === 0) return { datasets: [] };

    const data = [];
    const maxValue = Math.max(...confusionMatrix.flat());
    
    for (let i = 0; i < confusionMatrix.length; i++) {
      for (let j = 0; j < confusionMatrix[i].length; j++) {
        const value = confusionMatrix[i][j];
        if (value > 0) {
          // Calculate color intensity based on value
          const intensity = value / maxValue;
          const isDiagonal = i === j;
          
          // Use different colors for diagonal vs non-diagonal
          let backgroundColor;
          if (isDiagonal) {
            // Green for correct predictions (diagonal)
            backgroundColor = `rgba(76, 175, 80, ${intensity})`;
          } else {
            // Red for incorrect predictions (off-diagonal)
            backgroundColor = `rgba(244, 67, 54, ${intensity})`;
          }

          data.push({
            x: j,
            y: i,
            r: Math.max(5, Math.sqrt(value) * 2), // Size based on value
            value: value,
            backgroundColor: backgroundColor
          });
        }
      }
    }

    return {
      datasets: [{
        label: 'Confusion Matrix',
        data: data,
        backgroundColor: data.map(point => point.backgroundColor),
        borderColor: 'rgba(0, 0, 0, 0.1)',
        borderWidth: 1,
        pointRadius: data.map(point => point.r),
        pointHoverRadius: data.map(point => point.r + 2),
      }]
    };
  };

  const options: ChartOptions<'scatter'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: `Confusion Matrix Heatmap - ${selectedModel}`,
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            const point = context.raw;
            const trueClass = targetNames[point.y] || `Class ${point.y}`;
            const predClass = targetNames[point.x] || `Class ${point.x}`;
            return [
              `True: ${trueClass}`,
              `Predicted: ${predClass}`,
              `Count: ${point.value}`
            ];
          }
        }
      }
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        min: -0.5,
        max: targetNames.length - 0.5,
        ticks: {
          stepSize: 1,
          font: {
            size: 12
          },
          callback: function(value: any) {
            const index = Math.round(value);
            if (index >= 0 && index < targetNames.length) {
              return targetNames[index].substring(0, 20) + '...';
            }
            return '';
          }
        },
        title: {
          display: true,
          text: 'Predicted Class'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      y: {
        type: 'linear',
        min: -0.5,
        max: targetNames.length - 0.5,
        ticks: {
          stepSize: 1,
          font: {
            size: 12
          },
          callback: function(value: any) {
            const index = Math.round(value);
            if (index >= 0 && index < targetNames.length) {
              return targetNames[index].substring(0, 20) + '...';
            }
            return '';
          }
        },
        title: {
          display: true,
          text: 'True Class'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    }
  };

  // Create a table-based heatmap for better diagonal visualization
  const createTableHeatmap = () => {
    if (confusionMatrix.length === 0) return null;

    const maxValue = Math.max(...confusionMatrix.flat());
    
    return (
      <TableContainer component={Paper} sx={{ maxHeight: 800, overflow: 'auto' }}>
        <Table size="medium" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 'bold', backgroundColor: '#f5f5f5' }}>True\Pred</TableCell>
              {targetNames.map((name, index) => (
                <TableCell 
                  key={index} 
                  sx={{ 
                    fontWeight: 'bold', 
                    backgroundColor: '#f5f5f5',
                    fontSize: '0.9rem',
                    padding: '8px'
                  }}
                >
                  {name.substring(0, 18)}...
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {confusionMatrix.map((row, i) => (
              <TableRow key={i}>
                <TableCell 
                  sx={{ 
                    fontWeight: 'bold', 
                    backgroundColor: '#f5f5f5',
                    fontSize: '0.9rem',
                    padding: '8px'
                  }}
                >
                  {targetNames[i].substring(0, 18)}...
                </TableCell>
                {row.map((cell, j) => {
                  const intensity = cell / maxValue;
                  const isDiagonal = i === j;
                  const backgroundColor = isDiagonal 
                    ? `rgba(76, 175, 80, ${intensity})`
                    : `rgba(244, 67, 54, ${intensity})`;
                  
                  return (
                    <TableCell 
                      key={j} 
                      sx={{ 
                        backgroundColor,
                        color: intensity > 0.5 ? 'white' : 'black',
                        fontWeight: isDiagonal ? 'bold' : 'normal',
                        fontSize: '0.9rem',
                        padding: '8px',
                        textAlign: 'center',
                        border: isDiagonal ? '2px solid #2e7d32' : '1px solid #ddd'
                      }}
                    >
                      {cell}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  return (
    <Box>
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

      {confusionMatrix.length > 0 && (
        <Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Heatmap showing confusion matrix. Green cells on diagonal = correct predictions, red cells off-diagonal = incorrect predictions.
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Paper elevation={2} sx={{ p: 2 }}>
                <Typography variant="h6" sx={{ mb: 2 }}>Scatter Heatmap</Typography>
                <Box sx={{ height: 600, width: '100%' }}>
                  <Scatter data={createHeatmapData()} options={options} />
                </Box>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Paper elevation={2} sx={{ p: 2 }}>
                <Typography variant="h6" sx={{ mb: 2 }}>Table Heatmap (All 20 Classes)</Typography>
                {createTableHeatmap()}
              </Paper>
            </Grid>
          </Grid>

          <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ width: 20, height: 20, backgroundColor: 'rgba(76, 175, 80, 0.8)', borderRadius: '50%' }} />
              <Typography variant="body2">Correct Predictions (Diagonal)</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ width: 20, height: 20, backgroundColor: 'rgba(244, 67, 54, 0.8)', borderRadius: '50%' }} />
              <Typography variant="body2">Incorrect Predictions (Off-diagonal)</Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Darker colors = higher counts
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  );
} 