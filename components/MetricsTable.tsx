'use client';

import { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box,
  TableSortLabel,
  useTheme,
  useMediaQuery
} from '@mui/material';

interface ModelResult {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  cv_mean: number;
  cv_std: number;
  validation_accuracy?: number;
  training_time?: number;  // Add training time
}

interface MetricsTableProps {
  modelResults?: Record<string, ModelResult>;
  targetNames?: string[];
}

type SortField = 'model' | 'accuracy' | 'precision' | 'recall' | 'f1_score' | 'cv_mean' | 'validation_accuracy' | 'training_time';
type SortDirection = 'asc' | 'desc';

export default function MetricsTable({ modelResults, targetNames }: MetricsTableProps) {
  // Add null check for modelResults
  const [sortField, setSortField] = useState<SortField>('accuracy');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));

  if (!modelResults || Object.keys(modelResults).length === 0) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography color="text.secondary">
          Loading metrics data...
        </Typography>
      </Box>
    );
  }

  const formatPercentage = (value: number) => `${(value * 100).toFixed(2)}%`;
  const formatWithStd = (mean: number, std: number) => `${(mean * 100).toFixed(2)}% ± ${(std * 100).toFixed(2)}%`;

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const sortData = (data: [string, ModelResult][]) => {
    return data.sort((a, b) => {
      let aValue: any, bValue: any;
      
      switch (sortField) {
        case 'model':
          aValue = a[0].toLowerCase();
          bValue = b[0].toLowerCase();
          break;
        case 'accuracy':
          aValue = a[1].accuracy;
          bValue = b[1].accuracy;
          break;
        case 'precision':
          aValue = a[1].precision;
          bValue = b[1].precision;
          break;
        case 'recall':
          aValue = a[1].recall;
          bValue = b[1].recall;
          break;
        case 'f1_score':
          aValue = a[1].f1_score;
          bValue = b[1].f1_score;
          break;
        case 'cv_mean':
          aValue = a[1].cv_mean;
          bValue = b[1].cv_mean;
          break;
        case 'validation_accuracy':
          aValue = a[1].validation_accuracy || 0;
          bValue = b[1].validation_accuracy || 0;
          break;
        case 'training_time':
          aValue = a[1].training_time || 0;
          bValue = b[1].training_time || 0;
          break;
        default:
          return 0;
      }

      if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });
  };

  const sortedData = sortData(Object.entries(modelResults));

  const cellPadding = isSmallScreen ? '4px' : isMobile ? '6px' : '8px';
  const fontSize = isSmallScreen ? '0.7rem' : isMobile ? '0.8rem' : '0.875rem';
  const headerFontSize = isSmallScreen ? '0.75rem' : isMobile ? '0.85rem' : '0.875rem';

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ 
        fontSize: isSmallScreen ? '1.25rem' : isMobile ? '1.5rem' : '1.75rem',
        mb: isMobile ? 1 : 2
      }}>
        Model Performance Metrics
      </Typography>
      
      <TableContainer component={Paper} sx={{ 
        maxHeight: isSmallScreen ? 400 : isMobile ? 600 : 800,
        overflow: 'auto',
        '& .MuiTableCell-root': {
          padding: cellPadding,
          fontSize: fontSize
        }
      }}>
        <Table size={isSmallScreen ? "small" : "medium"} stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell sx={{ 
                fontWeight: 'bold', 
                backgroundColor: '#f5f5f5',
                fontSize: headerFontSize,
                padding: cellPadding,
                minWidth: isSmallScreen ? 80 : isMobile ? 100 : 120
              }}>
                <TableSortLabel
                  active={sortField === 'model'}
                  direction={sortField === 'model' ? sortDirection : 'asc'}
                  onClick={() => handleSort('model')}
                  sx={{ fontSize: headerFontSize }}
                >
                  Model
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ 
                fontWeight: 'bold', 
                backgroundColor: '#f5f5f5',
                fontSize: headerFontSize,
                padding: cellPadding,
                minWidth: isSmallScreen ? 60 : isMobile ? 70 : 80
              }}>
                <TableSortLabel
                  active={sortField === 'accuracy'}
                  direction={sortField === 'accuracy' ? sortDirection : 'asc'}
                  onClick={() => handleSort('accuracy')}
                  sx={{ fontSize: headerFontSize }}
                >
                  Accuracy
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ 
                fontWeight: 'bold', 
                backgroundColor: '#f5f5f5',
                fontSize: headerFontSize,
                padding: cellPadding,
                minWidth: isSmallScreen ? 60 : isMobile ? 70 : 80
              }}>
                <TableSortLabel
                  active={sortField === 'precision'}
                  direction={sortField === 'precision' ? sortDirection : 'asc'}
                  onClick={() => handleSort('precision')}
                  sx={{ fontSize: headerFontSize }}
                >
                  Precision
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ 
                fontWeight: 'bold', 
                backgroundColor: '#f5f5f5',
                fontSize: headerFontSize,
                padding: cellPadding,
                minWidth: isSmallScreen ? 60 : isMobile ? 70 : 80
              }}>
                <TableSortLabel
                  active={sortField === 'recall'}
                  direction={sortField === 'recall' ? sortDirection : 'asc'}
                  onClick={() => handleSort('recall')}
                  sx={{ fontSize: headerFontSize }}
                >
                  Recall
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ 
                fontWeight: 'bold', 
                backgroundColor: '#f5f5f5',
                fontSize: headerFontSize,
                padding: cellPadding,
                minWidth: isSmallScreen ? 60 : isMobile ? 70 : 80
              }}>
                <TableSortLabel
                  active={sortField === 'f1_score'}
                  direction={sortField === 'f1_score' ? sortDirection : 'asc'}
                  onClick={() => handleSort('f1_score')}
                  sx={{ fontSize: headerFontSize }}
                >
                  F1-Score
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ 
                fontWeight: 'bold', 
                backgroundColor: '#f5f5f5',
                fontSize: headerFontSize,
                padding: cellPadding,
                minWidth: isSmallScreen ? 80 : isMobile ? 90 : 100
              }}>
                <TableSortLabel
                  active={sortField === 'cv_mean'}
                  direction={sortField === 'cv_mean' ? sortDirection : 'asc'}
                  onClick={() => handleSort('cv_mean')}
                  sx={{ fontSize: headerFontSize }}
                >
                  CV Score
                </TableSortLabel>
              </TableCell>
              {sortedData.some(([_, result]) => result.validation_accuracy !== undefined) && (
                <TableCell sx={{ 
                  fontWeight: 'bold', 
                  backgroundColor: '#f5f5f5',
                  fontSize: headerFontSize,
                  padding: cellPadding,
                  minWidth: isSmallScreen ? 80 : isMobile ? 90 : 100
                }}>
                  <TableSortLabel
                    active={sortField === 'validation_accuracy'}
                    direction={sortField === 'validation_accuracy' ? sortDirection : 'asc'}
                    onClick={() => handleSort('validation_accuracy')}
                    sx={{ fontSize: headerFontSize }}
                  >
                    Val Acc
                  </TableSortLabel>
                </TableCell>
              )}
              {sortedData.some(([_, result]) => result.training_time !== undefined) && (
                <TableCell sx={{ 
                  fontWeight: 'bold', 
                  backgroundColor: '#f5f5f5',
                  fontSize: headerFontSize,
                  padding: cellPadding,
                  minWidth: isSmallScreen ? 60 : isMobile ? 70 : 80
                }}>
                  <TableSortLabel
                    active={sortField === 'training_time'}
                    direction={sortField === 'training_time' ? sortDirection : 'asc'}
                    onClick={() => handleSort('training_time')}
                    sx={{ fontSize: headerFontSize }}
                  >
                    Time (s)
                  </TableSortLabel>
                </TableCell>
              )}
            </TableRow>
          </TableHead>
          <TableBody>
            {sortedData.map(([modelName, result]) => (
              <TableRow key={modelName} hover>
                <TableCell sx={{ 
                  fontWeight: 'bold',
                  fontSize: fontSize,
                  padding: cellPadding,
                  minWidth: isSmallScreen ? 80 : isMobile ? 100 : 120
                }}>
                  {modelName}
                </TableCell>
                <TableCell sx={{ fontSize: fontSize, padding: cellPadding, minWidth: isSmallScreen ? 60 : isMobile ? 70 : 80 }}>
                  {formatPercentage(result.accuracy)}
                </TableCell>
                <TableCell sx={{ fontSize: fontSize, padding: cellPadding, minWidth: isSmallScreen ? 60 : isMobile ? 70 : 80 }}>
                  {formatPercentage(result.precision)}
                </TableCell>
                <TableCell sx={{ fontSize: fontSize, padding: cellPadding, minWidth: isSmallScreen ? 60 : isMobile ? 70 : 80 }}>
                  {formatPercentage(result.recall)}
                </TableCell>
                <TableCell sx={{ fontSize: fontSize, padding: cellPadding, minWidth: isSmallScreen ? 60 : isMobile ? 70 : 80 }}>
                  {formatPercentage(result.f1_score)}
                </TableCell>
                <TableCell sx={{ fontSize: fontSize, padding: cellPadding, minWidth: isSmallScreen ? 80 : isMobile ? 90 : 100 }}>
                  {formatWithStd(result.cv_mean, result.cv_std)}
                </TableCell>
                {sortedData.some(([_, r]) => r.validation_accuracy !== undefined) && (
                  <TableCell sx={{ fontSize: fontSize, padding: cellPadding, minWidth: isSmallScreen ? 80 : isMobile ? 90 : 100 }}>
                    {result.validation_accuracy !== undefined ? formatPercentage(result.validation_accuracy) : '-'}
                  </TableCell>
                )}
                {sortedData.some(([_, r]) => r.training_time !== undefined) && (
                  <TableCell sx={{ fontSize: fontSize, padding: cellPadding, minWidth: isSmallScreen ? 60 : isMobile ? 70 : 80 }}>
                    {result.training_time !== undefined ? result.training_time.toFixed(2) : '-'}
                  </TableCell>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
} 