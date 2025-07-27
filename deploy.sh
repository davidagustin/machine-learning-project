#!/bin/bash

echo "🚀 Deploying Machine Learning Project to Vercel"
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "⚠️  .env.local not found. Creating it..."
    ./setup-env.sh
fi

echo "📦 Building the project..."
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "🌐 Deploying to Vercel..."
echo "   This will open a browser window for authentication if needed."
echo ""

# Deploy to Vercel
vercel --prod

echo ""
echo "🎉 Deployment complete!"
echo "   Your app should now be live on Vercel!"
echo ""
echo "📋 Next Steps:"
echo "   1. Check your Vercel dashboard for the deployment URL"
echo "   2. Verify that the ML analysis is working"
echo "   3. Check that Redis caching is functioning"
echo "" 