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
  TableSortLabel
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
  modelResults: Record<string, ModelResult>;
  targetNames?: string[];
}

type SortField = 'model' | 'accuracy' | 'precision' | 'recall' | 'f1_score' | 'cv_mean' | 'validation_accuracy' | 'training_time';
type SortDirection = 'asc' | 'desc';

export default function MetricsTable({ modelResults, targetNames }: MetricsTableProps) {
  const [sortField, setSortField] = useState<SortField>('accuracy');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

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

  return (
    <Box>
      <Typography variant="h5" component="h2" gutterBottom>
        Model Performance Metrics
      </Typography>
      <TableContainer component={Paper} elevation={2}>
        <Table>
          <TableHead>
            <TableRow sx={{ backgroundColor: 'primary.main' }}>
              <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>
                <TableSortLabel
                  active={sortField === 'model'}
                  direction={sortField === 'model' ? sortDirection : 'asc'}
                  onClick={() => handleSort('model')}
                  sx={{ color: 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  Model
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ color: 'white', fontWeight: 'bold' }} align="right">
                <TableSortLabel
                  active={sortField === 'accuracy'}
                  direction={sortField === 'accuracy' ? sortDirection : 'asc'}
                  onClick={() => handleSort('accuracy')}
                  sx={{ color: 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  Accuracy
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ color: 'white', fontWeight: 'bold' }} align="right">
                <TableSortLabel
                  active={sortField === 'precision'}
                  direction={sortField === 'precision' ? sortDirection : 'asc'}
                  onClick={() => handleSort('precision')}
                  sx={{ color: 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  Precision
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ color: 'white', fontWeight: 'bold' }} align="right">
                <TableSortLabel
                  active={sortField === 'recall'}
                  direction={sortField === 'recall' ? sortDirection : 'asc'}
                  onClick={() => handleSort('recall')}
                  sx={{ color: 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  Recall
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ color: 'white', fontWeight: 'bold' }} align="right">
                <TableSortLabel
                  active={sortField === 'f1_score'}
                  direction={sortField === 'f1_score' ? sortDirection : 'asc'}
                  onClick={() => handleSort('f1_score')}
                  sx={{ color: 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  F1-Score
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ color: 'white', fontWeight: 'bold' }} align="right">
                <TableSortLabel
                  active={sortField === 'cv_mean'}
                  direction={sortField === 'cv_mean' ? sortDirection : 'asc'}
                  onClick={() => handleSort('cv_mean')}
                  sx={{ color: 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  CV Score
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ color: 'white', fontWeight: 'bold' }} align="right">
                <TableSortLabel
                  active={sortField === 'validation_accuracy'}
                  direction={sortField === 'validation_accuracy' ? sortDirection : 'asc'}
                  onClick={() => handleSort('validation_accuracy')}
                  sx={{ color: 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  Validation Acc.
                </TableSortLabel>
              </TableCell>
              <TableCell sx={{ color: 'white', fontWeight: 'bold' }} align="right">
                <TableSortLabel
                  active={sortField === 'training_time'}
                  direction={sortField === 'training_time' ? sortDirection : 'asc'}
                  onClick={() => handleSort('training_time')}
                  sx={{ color: 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  Training Time
                </TableSortLabel>
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sortedData.map(([modelName, metrics]) => (
              <TableRow key={modelName} hover>
                <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                  {modelName}
                </TableCell>
                <TableCell align="right">
                  <Typography variant="body2" color="text.secondary">
                    {formatPercentage(metrics.accuracy)}
                  </Typography>
                </TableCell>
                <TableCell align="right">
                  <Typography variant="body2" color="text.secondary">
                    {formatPercentage(metrics.precision)}
                  </Typography>
                </TableCell>
                <TableCell align="right">
                  <Typography variant="body2" color="text.secondary">
                    {formatPercentage(metrics.recall)}
                  </Typography>
                </TableCell>
                <TableCell align="right">
                  <Typography variant="body2" color="text.secondary">
                    {formatPercentage(metrics.f1_score)}
                  </Typography>
                </TableCell>
                <TableCell align="right">
                  <Typography variant="body2" color="text.secondary">
                    {formatWithStd(metrics.cv_mean, metrics.cv_std)}
                  </Typography>
                </TableCell>
                <TableCell align="right">
                  <Typography variant="body2" color="text.secondary">
                    {metrics.validation_accuracy ? formatPercentage(metrics.validation_accuracy) : 'N/A'}
                  </Typography>
                </TableCell>
                <TableCell align="right">
                  <Typography variant="body2" color="text.secondary">
                    {metrics.training_time ? `${metrics.training_time.toFixed(2)}s` : 'N/A'}
                  </Typography>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
} 