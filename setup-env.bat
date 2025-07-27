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

REM Create .env.local file
(
echo # Environment configuration
echo # Add any environment variables here as needed
) > .env.local

echo ✅ .env.local file created successfully!
echo.
echo 🔒 Security Notes:
echo    - .env.local is automatically ignored by git
echo    - Add any environment variables you need here
echo.
echo 📋 Next Steps:
echo    1. The ML processor will save results to static files
echo    2. Results will be served from the data directory
echo.
echo 🚀 You can now run: npm run dev
echo.
pause 