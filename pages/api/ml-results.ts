import { NextApiRequest, NextApiResponse } from 'next';
import fs from 'fs';
import path from 'path';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Path to the static data file
    const dataPath = path.join(process.cwd(), 'data', 'ml_results.json');
    
    // Check if the data file exists
    if (!fs.existsSync(dataPath)) {
      return res.status(404).json({ 
        error: 'ML results data not found. Please run the data generation script first.' 
      });
    }

    // Read the static data file
    const data = fs.readFileSync(dataPath, 'utf8');
    const mlResults = JSON.parse(data);

    // Add cache metadata to indicate this is static data
    const response = {
      ...mlResults,
      cache_metadata: {
        cached: true,
        cache_available: true,
        source: 'static_data' as const,
        data_file: 'data/ml_results.json',
        last_updated: mlResults.metadata?.computed_at || '2025-01-27T19:00:00Z'
      }
    };

    console.log('Serving ML results from static data file');
    
    res.status(200).json(response);
    
  } catch (error) {
    console.error('Error serving static ML results:', error);
    res.status(500).json({ 
      error: 'Failed to load ML results from static data',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
} 