@echo off
REM Setup script for environment variables (Windows)
echo 🔧 Setting up environment variables for Machine Learning Project
echo.

REM Check if .env.local already exists
if exist ".env.local" (
    echo ⚠️  .env.local already exists!
    echo    If you want to update it, please edit it manually.
    echo    Current content:
    echo    --------------------
    type .env.local
    echo    --------------------
    pause
    exit /b 0
)

echo 📝 Creating .env.local file...
echo.

REM Create .env.local with Vercel Redis configuration
(
echo # Redis Configuration for Vercel Redis
echo REDIS_URL=redis://default:af4tenGgBW1FIbhHF0O78NI3xYHs6fFv@redis-10393.c14.us-east-1-3.ec2.redns.redis-cloud.com:10393
echo.
echo # Optional: Local Redis fallback configuration
echo # REDIS_HOST=localhost
echo # REDIS_PORT=6379
echo # REDIS_PASSWORD=
) > .env.local

echo ✅ .env.local file created successfully!
echo.
echo 🔒 Security Notes:
echo    - .env.local is automatically ignored by git
echo    - Never commit your actual Redis credentials
echo    - The Redis URL is configured for Vercel Redis
echo.
echo 📋 Next Steps:
echo    1. The Redis URL is already configured for Vercel Redis
echo    2. For local development: You can change to redis://localhost:6379
echo    3. The ML processor will automatically cache results after each model completes
echo.
echo 🚀 You can now run: npm run dev
echo.
pause 