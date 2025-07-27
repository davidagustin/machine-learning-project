import { NextApiRequest, NextApiResponse } from 'next';
import { clearMLResultsCache, getCacheInfo } from '../../lib/redis';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const cacheInfo = await getCacheInfo();
    
    if (!cacheInfo.available) {
      return res.status(503).json({ 
        error: 'Cache service unavailable',
        message: 'Redis is not available'
      });
    }

    await clearMLResultsCache();
    
    return res.status(200).json({ 
      message: 'Cache cleared successfully',
      timestamp: new Date().toISOString(),
      cache_available: true
    });

  } catch (error) {
    console.error('Error clearing cache:', error);
    return res.status(500).json({ 
      error: 'Failed to clear cache',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
} 