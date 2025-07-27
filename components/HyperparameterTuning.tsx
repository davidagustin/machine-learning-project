'use client';

import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  useTheme,
  useMediaQuery
} from '@mui/material';
import { ExpandMore, Tune, TrendingUp, Science } from '@mui/icons-material';

interface HyperparameterTuningProps {
  hyperparameterTuning?: {
    [modelName: string]: {
      best_params: Record<string, any>;
      best_score: number;
      cv_results: {
        mean_test_score: number[];
        std_test_score: number[];
        params: Record<string, any>[];
      };
    };
  };
}

const HyperparameterTuning: React.FC<HyperparameterTuningProps> = ({ hyperparameterTuning }) => {
  const [expandedModel, setExpandedModel] = useState<string | false>(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));

  if (!hyperparameterTuning || Object.keys(hyperparameterTuning).length === 0) {
    return (
      <Card elevation={3}>
        <CardContent>
          <Typography variant="h5" component="div" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tune /> Hyperparameter Tuning
          </Typography>
          <Typography color="text.secondary">
            Loading hyperparameter tuning results...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const handleAccordionChange = (modelName: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedModel(isExpanded ? modelName : false);
  };

  // Sort models by best score
  const sortedModels = Object.entries(hyperparameterTuning).sort(
    ([, a], [, b]) => b.best_score - a.best_score
  );

  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h5" component="div" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Tune /> Hyperparameter Tuning Results
        </Typography>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Cross-validation results for optimized model parameters
        </Typography>

        <Grid container spacing={{ xs: 1, sm: 2 }}>
          {/* Summary Cards */}
          {sortedModels.map(([modelName, tuningData]) => (
            <Grid item xs={12} sm={6} md={4} key={modelName}>
              <Paper elevation={2} sx={{ p: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Science /> {modelName}
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h4" color="primary" sx={{ fontWeight: 'bold' }}>
                    {(tuningData.best_score * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Best CV Score
                  </Typography>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Best Parameters:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {Object.entries(tuningData.best_params).map(([param, value]) => (
                      <Chip
                        key={param}
                        label={`${param}: ${value}`}
                        size="small"
                        variant="outlined"
                        color="primary"
                        sx={{
                          fontSize: { xs: '0.625rem', sm: '0.75rem' },
                          '& .MuiChip-label': {
                            wordBreak: 'break-word',
                            textAlign: 'center'
                          }
                        }}
                      />
                    ))}
                  </Box>
                </Box>

                <LinearProgress
                  variant="determinate"
                  value={tuningData.best_score * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Paper>
            </Grid>
          ))}
        </Grid>

        {/* Detailed Results */}
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TrendingUp /> Detailed Cross-Validation Results
          </Typography>
          
          {sortedModels.map(([modelName, tuningData]) => (
            <Accordion
              key={modelName}
              expanded={expandedModel === modelName}
              onChange={handleAccordionChange(modelName)}
              sx={{ mb: 1 }}
            >
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    {modelName}
                  </Typography>
                  <Chip
                    label={`${(tuningData.best_score * 100).toFixed(1)}% CV Score`}
                    color="primary"
                    variant="filled"
                  />
                </Box>
              </AccordionSummary>
              
              <AccordionDetails>
                <Grid container spacing={2}>
                  {/* Best Parameters */}
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Best Parameters:
                    </Typography>
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Parameter</TableCell>
                            <TableCell>Value</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(tuningData.best_params).map(([param, value]) => (
                            <TableRow key={param}>
                              <TableCell component="th" scope="row">
                                <code>{param}</code>
                              </TableCell>
                              <TableCell>
                                <code>{String(value)}</code>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Grid>

                  {/* CV Results Summary */}
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Cross-Validation Results:
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Mean Score: {((tuningData.cv_results.mean_test_score.reduce((a, b) => a + b, 0) / tuningData.cv_results.mean_test_score.length) * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Std Score: {((tuningData.cv_results.std_test_score.reduce((a, b) => a + b, 0) / tuningData.cv_results.std_test_score.length) * 100).toFixed(2)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Total Combinations: {tuningData.cv_results.params.length}
                      </Typography>
                    </Box>

                    {/* Score Distribution */}
                    <Typography variant="body2" gutterBottom>
                      Score Distribution:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {tuningData.cv_results.mean_test_score.slice(0, 5).map((score, index) => (
                        <Chip
                          key={index}
                          label={`${(score * 100).toFixed(1)}%`}
                          size="small"
                          variant="outlined"
                          color={score === tuningData.best_score ? "primary" : "default"}
                        />
                      ))}
                      {tuningData.cv_results.mean_test_score.length > 5 && (
                        <Chip
                          label={`+${tuningData.cv_results.mean_test_score.length - 5} more`}
                          size="small"
                          variant="outlined"
                        />
                      )}
                    </Box>
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default HyperparameterTuning; 