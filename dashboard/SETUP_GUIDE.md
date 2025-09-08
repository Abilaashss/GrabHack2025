# Grab Credit Score Dashboard - Setup Guide

This guide will help you set up and integrate the dashboard with your existing ML models and data.

## Quick Start

### 1. Install Dependencies
```bash
cd dashboard
npm install
```

### 2. Start Development Server
```bash
npm run dev
```

### 3. Access the Dashboard
- Open http://localhost:3000
- Choose between User Portal or Admin Dashboard

## Integration with Your ML Models

### Step 1: Copy Model Files
```bash
# Copy your trained models to the dashboard's public directory
mkdir -p dashboard/public/models
cp models/saved_models/* dashboard/public/models/

# Copy visualization plots
mkdir -p dashboard/public/plots  
cp results/plots/* dashboard/public/plots/
```

### Step 2: Update Data Service
Edit `dashboard/lib/dataService.ts`:

```typescript
// Replace mock data with your CSV files
export const loadData = async () => {
  const driversResponse = await fetch('/api/data/drivers.csv')
  const driversText = await driversResponse.text()
  const driversData = Papa.parse(driversText, { header: true }).data
  
  const merchantsResponse = await fetch('/api/data/merchants.csv')
  const merchantsText = await merchantsResponse.text()
  const merchantsData = Papa.parse(merchantsText, { header: true }).data
  
  return { driversData, merchantsData }
}
```

### Step 3: Connect ML Models
Edit `dashboard/lib/modelService.ts`:

```typescript
export const predictCreditScore = async (user: any, modelName: string) => {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user, model: modelName })
  })
  
  const result = await response.json()
  return result.creditScore
}
```

## Backend API Setup (Optional)

If you want to create a backend API to serve your models:

### 1. Create FastAPI Backend
```python
# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

# Enable CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
models = {
    'random_forest': joblib.load('../models/saved_models/drivers_random_forest.pkl'),
    'gradient_boosting': joblib.load('../models/saved_models/drivers_gradient_boosting.pkl'),
    # Add other models...
}

@app.post("/predict")
async def predict_credit_score(request: dict):
    user_data = request['user']
    model_name = request['model']
    
    # Prepare features
    features = prepare_features(user_data)
    
    # Make prediction
    model = models[model_name]
    prediction = model.predict([features])[0]
    
    return {"creditScore": float(prediction)}

def prepare_features(user_data):
    # Convert user data to model features
    # This depends on your model's expected input format
    pass
```

### 2. Start Backend Server
```bash
cd api
pip install fastapi uvicorn
uvicorn main:app --reload --port 8000
```

### 3. Update Frontend Configuration
```bash
# dashboard/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Data Integration Options

### Option 1: Static CSV Files
```bash
# Copy CSV files to public directory
cp data/*.csv dashboard/public/data/
```

### Option 2: Database Integration
```typescript
// lib/database.ts
import { Pool } from 'pg'

const pool = new Pool({
  connectionString: process.env.DATABASE_URL
})

export const getUsers = async (userType: string) => {
  const result = await pool.query(
    'SELECT * FROM users WHERE partner_type = $1',
    [userType]
  )
  return result.rows
}
```

### Option 3: API Integration
```typescript
// lib/dataService.ts
export const searchUsers = async (searchTerm: string, userType: string) => {
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/users/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ searchTerm, userType })
  })
  
  return response.json()
}
```

## Customization Guide

### 1. Branding and Colors
Edit `tailwind.config.js`:
```javascript
theme: {
  extend: {
    colors: {
      grab: {
        50: '#f0fdf4',   // Light green
        500: '#00b14f',  // Grab green
        900: '#14532d',  // Dark green
      }
    }
  }
}
```

### 2. Adding New Parameters
To add new user parameters:

1. Update the parameter display in `components/user/ParametersView.tsx`
2. Add parameter labels in the `getParameterLabel` function
3. Update formatting logic in `formatValue` function

### 3. Custom Visualizations
To add new charts:

1. Create new chart components using Recharts
2. Add them to the appropriate dashboard sections
3. Update the data processing logic

### 4. Authentication
To add user authentication:

```bash
npm install next-auth
```

```typescript
// pages/api/auth/[...nextauth].ts
import NextAuth from 'next-auth'
import CredentialsProvider from 'next-auth/providers/credentials'

export default NextAuth({
  providers: [
    CredentialsProvider({
      name: 'credentials',
      credentials: {
        partnerId: { label: 'Partner ID', type: 'text' },
        userType: { label: 'User Type', type: 'text' }
      },
      async authorize(credentials) {
        // Validate user credentials
        const user = await validateUser(credentials)
        return user ? { id: user.partner_id, ...user } : null
      }
    })
  ]
})
```

## Production Deployment

### 1. Build the Application
```bash
npm run build
```

### 2. Environment Variables
Set production environment variables:
```bash
NEXT_PUBLIC_API_URL=https://your-api-domain.com
NEXTAUTH_SECRET=your-production-secret
```

### 3. Deploy Options

#### Vercel (Recommended)
```bash
npm install -g vercel
vercel --prod
```

#### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

#### Traditional Server
```bash
# Build and start
npm run build
npm start

# Or use PM2 for process management
npm install -g pm2
pm2 start npm --name "grab-dashboard" -- start
```

## Troubleshooting

### Common Issues

1. **Models not loading**: Ensure model files are in the correct format and path
2. **CORS errors**: Configure CORS properly in your backend API
3. **Data not displaying**: Check CSV file format and column names
4. **Performance issues**: Implement data pagination and caching

### Debug Mode
```bash
# Enable debug logging
DEBUG=* npm run dev
```

### Performance Monitoring
```typescript
// Add to your components
import { useEffect } from 'react'

useEffect(() => {
  console.time('Component Load Time')
  return () => console.timeEnd('Component Load Time')
}, [])
```

## Support

For issues and questions:
1. Check the console for error messages
2. Verify all file paths and API endpoints
3. Ensure all dependencies are installed correctly
4. Check that your data format matches the expected schema

## Next Steps

1. **Security**: Implement proper authentication and authorization
2. **Performance**: Add caching and optimize data loading
3. **Features**: Add more advanced analytics and reporting
4. **Mobile**: Enhance mobile responsiveness
5. **Real-time**: Consider WebSocket integration for live updates