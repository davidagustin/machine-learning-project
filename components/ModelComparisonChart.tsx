'use client';

import { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { useTheme, useMediaQuery } from '@mui/material';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
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
}

interface ModelComparisonChartProps {
  modelResults?: { [key: string]: ModelResult };
}

export default function ModelComparisonChart({ modelResults }: ModelComparisonChartProps) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));
  
  // Add null check for modelResults
  if (!modelResults || Object.keys(modelResults).length === 0) {
    return (
      <div style={{ 
        height: isSmallScreen ? '300px' : isMobile ? '350px' : '400px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center' 
      }}>
        <p>Loading model comparison data...</p>
      </div>
    );
  }

  const modelNames = Object.keys(modelResults);
  const metrics = ['accuracy', 'precision', 'recall', 'f1_score'];

  const data = {
    labels: modelNames,
    datasets: metrics.map((metric, index) => ({
      label: metric.charAt(0).toUpperCase() + metric.slice(1).replace('_', ' '),
      data: modelNames.map(name => modelResults[name][metric as keyof ModelResult] as number),
      backgroundColor: [
        'rgba(255, 99, 132, 0.8)',
        'rgba(54, 162, 235, 0.8)',
        'rgba(255, 206, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)',
      ][index],
      borderColor: [
        'rgba(255, 99, 132, 1)',
        'rgba(54, 162, 235, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)',
      ][index],
      borderWidth: 1,
    })),
  };

  const options: ChartOptions<'bar'> = {
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
        text: 'Model Performance Metrics Comparison',
        font: {
          size: isSmallScreen ? 14 : isMobile ? 15 : 16
        }
      },
    },
    scales: {
      x: {
        ticks: {
          font: {
            size: isSmallScreen ? 10 : isMobile ? 11 : 12
          }
        }
      },
      y: {
        ticks: {
          font: {
            size: isSmallScreen ? 10 : isMobile ? 11 : 12
          }
        }
      }
    }
  };

  return (
    <div>
      <h2 style={{ 
        fontSize: isSmallScreen ? '1.125rem' : isMobile ? '1.25rem' : '1.75rem',
        marginBottom: isSmallScreen ? '0.5rem' : isMobile ? '1rem' : '1.5rem'
      }}>
        Model Performance Comparison
      </h2>
      <div style={{ 
        height: isSmallScreen ? '250px' : isMobile ? '300px' : '400px',
        width: '100%'
      }}>
        <Bar data={data} options={options} />
      </div>
    </div>
  );
} 