#!/bin/bash

# Setup script for environment variables
echo "🔧 Setting up environment variables for Machine Learning Project"
echo ""

# Check if .env.local already exists
if [ -f ".env.local" ]; then
    echo "⚠️  .env.local already exists!"
    echo "   If you want to update it, please edit it manually."
    echo "   Current content:"
    echo "   --------------------"
    cat .env.local
    echo "   --------------------"
    exit 0
fi

echo "📝 Creating .env.local file..."
echo ""

# Create .env.local file
cat > .env.local << 'EOF'
# Environment configuration
# Add any environment variables here as needed
EOF

echo "✅ .env.local file created successfully!"
echo ""
echo "🔒 Security Notes:"
echo "   - .env.local is automatically ignored by git"
echo "   - Add any environment variables you need here"
echo ""
echo "📋 Next Steps:"
echo "   1. The ML processor will save results to static files"
echo "   2. Results will be served from the data directory"
echo ""
echo "🚀 You can now run: npm run dev" 