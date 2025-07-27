import Redis from 'ioredis';

// Redis configuration for Vercel Redis
const getRedisConfig = () => {
  // If REDIS_URL is provided (Vercel Redis), use it
  if (process.env.REDIS_URL) {
    return {
      url: process.env.REDIS_URL,
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3,
      lazyConnect: true,
    };
  }
  
  // Fallback to local Redis configuration
  return {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    password: process.env.REDIS_PASSWORD,
    retryDelayOnFailover: 100,
    maxRetriesPerRequest: 3,
    lazyConnect: true,
  };
};

const redis = new Redis(getRedisConfig());

// Cache configuration
const CACHE_TTL = 3600; // 1 hour in seconds
const ML_RESULTS_CACHE_KEY = 'ml_analysis_results';

// Cache utility functions
export const cacheMLResults = async (results: any): Promise<void> => {
  try {
    await redis.setex(ML_RESULTS_CACHE_KEY, CACHE_TTL, JSON.stringify(results));
    console.log('ML results cached successfully');
  } catch (error) {
    console.error('Error caching ML results:', error);
  }
};

export const getCachedMLResults = async (): Promise<any | null> => {
  try {
    const cached = await redis.get(ML_RESULTS_CACHE_KEY);
    if (cached) {
      console.log('ML results retrieved from cache');
      return JSON.parse(cached);
    }
    return null;
  } catch (error) {
    console.error('Error retrieving cached ML results:', error);
    return null;
  }
};

export const clearMLResultsCache = async (): Promise<void> => {
  try {
    await redis.del(ML_RESULTS_CACHE_KEY);
    console.log('ML results cache cleared');
  } catch (error) {
    console.error('Error clearing ML results cache:', error);
  }
};

export const isCacheAvailable = async (): Promise<boolean> => {
  try {
    await redis.ping();
    return true;
  } catch (error) {
    console.error('Redis not available:', error);
    return false;
  }
};

export const getCacheInfo = async (): Promise<{ available: boolean }> => {
  try {
    await redis.ping();
    return { available: true };
  } catch (error) {
    return { available: false };
  }
};

export default redis; 