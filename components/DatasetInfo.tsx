'use client';

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  Paper
} from '@mui/material';
import { DataUsage, TextFields, Storage, Timeline } from '@mui/icons-material';

interface DatasetInfoProps {
  datasetInfo: {
    total_samples: number;
    num_classes: number;
    class_distribution: Record<string, number>;
    avg_text_length: number;
    min_text_length: number;
    max_text_length: number;
    class_names: string[];
  };
  vectorizationInfo?: {
    vocabulary_size: number;
    max_features: number;
    ngram_range: number[];
    min_df: number;
    max_df: number;
    avg_features_per_document: number;
    top_features: [string, number][];
  };
  dataSplitInfo?: {
    total_samples: number;
    train_samples: number;
    val_samples: number;
    test_samples: number;
    train_percentage: number;
    val_percentage: number;
    test_percentage: number;
  };
}

const DatasetInfo: React.FC<DatasetInfoProps> = ({ datasetInfo, vectorizationInfo, dataSplitInfo }) => {
  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h5" component="div" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <DataUsage /> Dataset Information
        </Typography>
        <Grid container spacing={2}>
          {/* Basic Dataset Stats */}
          <Grid item xs={12} md={6}>
            <Paper elevation={1} sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Storage /> Basic Statistics
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText 
                    primary="Total Samples" 
                    secondary={datasetInfo.total_samples.toLocaleString()} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Number of Classes" 
                    secondary={datasetInfo.num_classes} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Average Text Length" 
                    secondary={`${datasetInfo.avg_text_length.toFixed(1)} words`} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Text Length Range" 
                    secondary={`${datasetInfo.min_text_length} - ${datasetInfo.max_text_length} words`} 
                  />
                </ListItem>
              </List>
            </Paper>
          </Grid>

          {/* Class Distribution */}
          <Grid item xs={12} md={6}>
            <Paper elevation={1} sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Class Distribution
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {Object.entries(datasetInfo.class_distribution).map(([className, count]) => (
                  <Chip
                    key={className}
                    label={`${className}: ${count}`}
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Paper>
          </Grid>

          {/* Vectorization Information */}
          {vectorizationInfo && (
            <Grid item xs={12} md={6}>
              <Paper elevation={1} sx={{ p: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TextFields /> Text Vectorization (TF-IDF)
                </Typography>
                <List dense>
                  <ListItem><ListItemText primary={`Vocabulary Size: ${vectorizationInfo.vocabulary_size}`}/></ListItem>
                  <ListItem><ListItemText primary={`Max Features: ${vectorizationInfo.max_features}`}/></ListItem>
                  <ListItem><ListItemText primary={`N-gram Range: ${vectorizationInfo.ngram_range.join('-')}`}/></ListItem>
                  <ListItem><ListItemText primary={`Min Document Frequency: ${vectorizationInfo.min_df}`}/></ListItem>
                  <ListItem><ListItemText primary={`Max Document Frequency: ${vectorizationInfo.max_df}`}/></ListItem>
                  <ListItem><ListItemText primary={`Avg. Features per Document: ${vectorizationInfo.avg_features_per_document.toFixed(2)}`}/></ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Top 20 Features:"
                      secondary={
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                          {vectorizationInfo.top_features.map(([feature, score]) => (
                            <Chip key={feature} label={`${feature} (${score.toFixed(2)})`} size="small" />
                          ))}
                        </Box>
                      }
                    />
                  </ListItem>
                </List>
              </Paper>
            </Grid>
          )}

          {/* Data Split Information */}
          {dataSplitInfo && (
            <Grid item xs={12} md={6}>
              <Paper elevation={1} sx={{ p: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Timeline /> Data Split
                </Typography>
                <List dense>
                  <ListItem><ListItemText primary={`Total Samples: ${dataSplitInfo.total_samples}`}/></ListItem>
                  <ListItem><ListItemText primary={`Training Set: ${dataSplitInfo.train_samples} (${dataSplitInfo.train_percentage}%)`}/></ListItem>
                  <ListItem><ListItemText primary={`Validation Set: ${dataSplitInfo.val_samples} (${dataSplitInfo.val_percentage}%)`}/></ListItem>
                  <ListItem><ListItemText primary={`Test Set: ${dataSplitInfo.test_samples} (${dataSplitInfo.test_percentage}%)`}/></ListItem>
                </List>
              </Paper>
            </Grid>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default DatasetInfo; 