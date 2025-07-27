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
  TableRow,
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
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));

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
            r: Math.max(3, Math.sqrt(value) * 1.5), // Smaller size for better fit
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
          size: isMobile ? 14 : 16
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
            size: isSmallScreen ? 10 : isMobile ? 12 : 14
          },
          maxRotation: 45,
          minRotation: 45,
          padding: 8,
          callback: function(value: any) {
            const index = Math.round(value);
            if (index >= 0 && index < targetNames.length) {
              const maxLength = isSmallScreen ? 15 : isMobile ? 18 : 25;
              const label = targetNames[index];
              if (label.length > maxLength) {
                // Try to keep the most important part (usually the last part)
                const parts = label.split('.');
                if (parts.length > 1) {
                  // Keep the last part and truncate if needed
                  const lastPart = parts[parts.length - 1];
                  if (lastPart.length <= maxLength) {
                    return lastPart;
                  } else {
                    return lastPart.substring(0, maxLength - 3) + '...';
                  }
                } else {
                  return label.substring(0, maxLength - 3) + '...';
                }
              }
              return label;
            }
            return '';
          }
        },
        title: {
          display: true,
          text: 'Predicted Class',
          font: {
            size: isMobile ? 14 : 16
          },
          padding: {
            top: 10
          }
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
            size: isSmallScreen ? 10 : isMobile ? 12 : 14
          },
          padding: 8,
          callback: function(value: any) {
            const index = Math.round(value);
            if (index >= 0 && index < targetNames.length) {
              const maxLength = isSmallScreen ? 15 : isMobile ? 18 : 25;
              const label = targetNames[index];
              if (label.length > maxLength) {
                // Try to keep the most important part (usually the last part)
                const parts = label.split('.');
                if (parts.length > 1) {
                  // Keep the last part and truncate if needed
                  const lastPart = parts[parts.length - 1];
                  if (lastPart.length <= maxLength) {
                    return lastPart;
                  } else {
                    return lastPart.substring(0, maxLength - 3) + '...';
                  }
                } else {
                  return label.substring(0, maxLength - 3) + '...';
                }
              }
              return label;
            }
            return '';
          }
        },
        title: {
          display: true,
          text: 'True Class',
          font: {
            size: isMobile ? 14 : 16
          },
          padding: {
            bottom: 10
          }
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
    const cellPadding = isSmallScreen ? '4px' : isMobile ? '6px' : '8px';
    const fontSize = isSmallScreen ? '0.7rem' : isMobile ? '0.8rem' : '0.9rem';
    const labelMaxLength = isSmallScreen ? 12 : isMobile ? 15 : 20;
    
    // Helper function for smart label truncation
    const truncateLabel = (label: string, maxLength: number) => {
      if (label.length <= maxLength) return label;
      
      const parts = label.split('.');
      if (parts.length > 1) {
        const lastPart = parts[parts.length - 1];
        if (lastPart.length <= maxLength) {
          return lastPart;
        } else {
          return lastPart.substring(0, maxLength - 3) + '...';
        }
      } else {
        return label.substring(0, maxLength - 3) + '...';
      }
    };
    
    return (
      <TableContainer 
        component={Paper} 
        sx={{ 
          maxHeight: isSmallScreen ? 400 : isMobile ? 600 : 800, 
          overflow: 'auto',
          '& .MuiTableCell-root': {
            padding: cellPadding,
            fontSize: fontSize
          }
        }}
      >
        <Table size={isSmallScreen ? "small" : "medium"} stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell sx={{ 
                fontWeight: 'bold', 
                backgroundColor: '#f5f5f5',
                fontSize: fontSize,
                padding: cellPadding,
                minWidth: isSmallScreen ? 60 : isMobile ? 80 : 100
              }}>
                True\Pred
              </TableCell>
              {targetNames.map((name, index) => (
                <TableCell 
                  key={index} 
                  sx={{ 
                    fontWeight: 'bold', 
                    backgroundColor: '#f5f5f5',
                    fontSize: fontSize,
                    padding: cellPadding,
                    minWidth: isSmallScreen ? 40 : isMobile ? 50 : 60
                  }}
                >
                  {truncateLabel(name, labelMaxLength)}
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
                    fontSize: fontSize,
                    padding: cellPadding,
                    minWidth: isSmallScreen ? 60 : isMobile ? 80 : 100
                  }}
                >
                  {truncateLabel(targetNames[i], labelMaxLength)}
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
                        fontSize: fontSize,
                        padding: cellPadding,
                        textAlign: 'center',
                        border: isDiagonal ? '2px solid #2e7d32' : '1px solid #ddd',
                        minWidth: isSmallScreen ? 40 : isMobile ? 50 : 60
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
          
          <Grid container spacing={isMobile ? 2 : 3}>
            <Grid item xs={12}>
              <Paper elevation={2} sx={{ p: isMobile ? 1 : 2 }}>
                <Typography variant="h6" sx={{ mb: 2, fontSize: isMobile ? '1.1rem' : '1.25rem' }}>
                  Scatter Heatmap
                </Typography>
                <Box sx={{ 
                  height: isSmallScreen ? 500 : isMobile ? 600 : 700, 
                  width: '100%',
                  minHeight: 400
                }}>
                  <Scatter data={createHeatmapData()} options={options} />
                </Box>
              </Paper>
            </Grid>
            
            <Grid item xs={12}>
              <Paper elevation={2} sx={{ p: isMobile ? 1 : 2 }}>
                <Typography variant="h6" sx={{ mb: 2, fontSize: isMobile ? '1.1rem' : '1.25rem' }}>
                  Table Heatmap (All 20 Classes)
                </Typography>
                {createTableHeatmap()}
              </Paper>
            </Grid>
          </Grid>

          <Box sx={{ 
            mt: 2, 
            display: 'flex', 
            gap: isMobile ? 1 : 2, 
            alignItems: 'center', 
            flexWrap: 'wrap',
            flexDirection: isSmallScreen ? 'column' : 'row'
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ 
                width: isSmallScreen ? 16 : 20, 
                height: isSmallScreen ? 16 : 20, 
                backgroundColor: 'rgba(76, 175, 80, 0.8)', 
                borderRadius: '50%' 
              }} />
              <Typography variant="body2" sx={{ fontSize: isSmallScreen ? '0.8rem' : '0.875rem' }}>
                Correct Predictions (Diagonal)
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ 
                width: isSmallScreen ? 16 : 20, 
                height: isSmallScreen ? 16 : 20, 
                backgroundColor: 'rgba(244, 67, 54, 0.8)', 
                borderRadius: '50%' 
              }} />
              <Typography variant="body2" sx={{ fontSize: isSmallScreen ? '0.8rem' : '0.875rem' }}>
                Incorrect Predictions (Off-diagonal)
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: isSmallScreen ? '0.8rem' : '0.875rem' }}>
              Darker colors = higher counts
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  );
} 