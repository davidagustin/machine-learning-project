import { NextApiRequest, NextApiResponse } from 'next';
import { exec } from 'child_process';
import path from 'path';
import { promisify } from 'util';
import { 
  getCachedMLResults, 
  cacheMLResults, 
  isCacheAvailable,
  getCacheInfo
} from '../../lib/redis';

const execAsync = promisify(exec);

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    // Check if Redis cache is available
    const cacheInfo = await getCacheInfo();
    
    if (cacheInfo.available) {
      // Try to get cached results first
      const cachedResults = await getCachedMLResults();
      
      if (cachedResults) {
        console.log('Serving ML results from cache');
        return res.status(200).json({
          ...cachedResults,
          cache_metadata: {
            cached: true,
            cache_available: true,
            source: 'redis_cache'
          }
        });
      }
    }

    // If no cache or no cached results, run the ML analysis
    console.log('Running ML analysis...');
    
    const scriptPath = path.join(process.cwd(), 'scripts', 'ml_processor.py');
    
    // Use virtual environment Python if available, otherwise fall back to system Python
    let pythonCommand = 'python3';
    
    // Check if virtual environment exists and use it
    const venvPythonPath = path.join(process.cwd(), 'venv', 'bin', 'python');
    try {
      const { stdout: pythonVersion } = await execAsync(`${venvPythonPath} --version`);
      console.log('Using virtual environment Python:', pythonVersion.trim());
      pythonCommand = venvPythonPath;
    } catch {
      // Fall back to system Python
      try {
        const { stdout: pythonVersion } = await execAsync('python3 --version');
        console.log('Using system python3:', pythonVersion.trim());
        pythonCommand = 'python3';
      } catch {
        try {
          const { stdout: pythonVersion } = await execAsync('python --version');
          console.log('Using system python:', pythonVersion.trim());
          pythonCommand = 'python';
        } catch {
          console.log('Python not found, using python3 as fallback');
        }
      }
    }
    
    const { stdout, stderr } = await execAsync(`${pythonCommand} ${scriptPath}`, {
      timeout: 300000, // 5 minutes timeout
      cwd: process.cwd(),
      env: {
        ...process.env,
        PYTHONPATH: process.cwd(),
        PYTHONUNBUFFERED: '1',
        REDIS_URL: process.env.REDIS_URL || 'redis://localhost:6379'
      }
    });

    if (stderr) {
      console.error('Python script stderr:', stderr);
    }

    // Clean stdout to get only the JSON output
    const lines = stdout.trim().split('\n');
    const jsonLine = lines[lines.length - 1]; // Get the last line which should be JSON
    
    let results;
    try {
      results = JSON.parse(jsonLine);
    } catch (parseError) {
      console.error('Error parsing Python output:', parseError);
      console.error('Raw output:', stdout);
      console.error('Attempted to parse:', jsonLine);
      return res.status(500).json({ 
        error: 'Failed to parse ML results',
        details: parseError instanceof Error ? parseError.message : 'Unknown error'
      });
    }

    // Cache the results if Redis is available
    if (cacheInfo.available) {
      await cacheMLResults(results);
    }

    // Return results with cache metadata
    return res.status(200).json({
      ...results,
      cache_metadata: {
        cached: false,
        cache_available: cacheInfo.available,
        source: 'fresh_analysis'
      }
    });

  } catch (error) {
    console.error('Error in ML results API:', error);
    
    // Check if it's a timeout error
    if (error instanceof Error && error.message.includes('timeout')) {
      return res.status(408).json({ 
        error: 'Request timeout',
        message: 'ML analysis took too long to complete'
      });
    }

    return res.status(500).json({ 
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
} 