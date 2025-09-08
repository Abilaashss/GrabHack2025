#!/bin/bash

echo "🚀 Setting up Grab Credit Score Dashboard..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18+ is required. Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js $(node -v) detected"

# Install dependencies
echo "📦 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Create environment file
if [ ! -f ".env.local" ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env.local
    echo "✅ Environment file created (.env.local)"
    echo "📝 Please update .env.local with your configuration"
fi

# Create public directories for data and models
echo "📁 Creating public directories..."
mkdir -p public/data
mkdir -p public/models  
mkdir -p public/plots

# Copy data files if they exist in parent directory
if [ -d "../data" ]; then
    echo "📊 Copying data files..."
    cp ../data/*.csv public/data/ 2>/dev/null || echo "⚠️  No CSV files found in ../data"
fi

# Copy model files if they exist
if [ -d "../models/saved_models" ]; then
    echo "🤖 Copying model files..."
    cp ../models/saved_models/* public/models/ 2>/dev/null || echo "⚠️  No model files found in ../models/saved_models"
fi

# Copy plot files if they exist
if [ -d "../results/plots" ]; then
    echo "📈 Copying visualization plots..."
    cp ../results/plots/* public/plots/ 2>/dev/null || echo "⚠️  No plot files found in ../results/plots"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Update .env.local with your configuration"
echo "2. Run 'npm run dev' to start the development server"
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "📚 For detailed integration guide, see SETUP_GUIDE.md"
echo ""

# Ask if user wants to start the dev server
read -p "🚀 Would you like to start the development server now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🌟 Starting development server..."
    npm run dev
fi