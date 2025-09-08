# Grab Credit Score Dashboard

A comprehensive Next.js dashboard for Grab's credit scoring system, providing interfaces for both users (drivers/merchants) and administrators.

## Features

### User Portal (Drivers & Merchants)
- **Credit Score Visualization**: Interactive credit score display with circular progress indicators
- **Parameter Analysis**: Detailed view of all factors affecting credit scores
- **Model Comparison**: Compare predictions across different ML models
- **Performance Metrics**: Track key performance indicators and trends
- **Responsive Design**: Optimized for desktop and mobile devices

### Admin Dashboard
- **System Overview**: Real-time statistics and system health monitoring
- **User Search**: Advanced search and filtering for all users
- **Analytics Dashboard**: Comprehensive analytics with interactive charts
- **Model Management**: Monitor and manage ML model performance
- **Visualizations**: Access to all model performance charts and plots

## Technology Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **Styling**: Tailwind CSS with custom green color palette
- **Charts**: Recharts for data visualization
- **Animations**: Framer Motion for smooth transitions
- **Icons**: Heroicons for consistent iconography

## Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

1. Navigate to the dashboard directory:
```bash
cd dashboard
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Start the development server:
```bash
npm run dev
# or
yarn dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

```
dashboard/
├── app/                    # Next.js app directory
│   ├── admin/             # Admin dashboard pages
│   ├── user/              # User portal pages
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page
├── components/            # React components
│   ├── admin/             # Admin-specific components
│   └── user/              # User-specific components
├── lib/                   # Utility libraries
│   ├── dataService.ts     # Data fetching and management
│   └── modelService.ts    # ML model integration
├── public/                # Static assets
└── package.json           # Dependencies and scripts
```

## Data Integration

The dashboard currently uses mock data for demonstration. To integrate with your actual data:

1. **Update Data Service** (`lib/dataService.ts`):
   - Replace mock data with API calls to your backend
   - Update CSV parsing logic for your actual data files

2. **Model Integration** (`lib/modelService.ts`):
   - Connect to your trained models in `models/saved_models/`
   - Implement actual prediction endpoints

3. **Visualization Assets**:
   - Copy plots from `results/plots/` to `public/plots/`
   - Update image paths in components

## Customization

### Color Scheme
The dashboard uses a green-focused color palette. Modify `tailwind.config.js` to customize:

```javascript
colors: {
  grab: {
    500: '#00b14f', // Primary green
    // ... other shades
  }
}
```

### Adding New Features
1. Create new components in the appropriate directory
2. Add routes in the `app/` directory
3. Update navigation in dashboard components

## Deployment

### Build for Production
```bash
npm run build
npm start
```

### Environment Variables
Create a `.env.local` file for environment-specific configuration:

```env
NEXT_PUBLIC_API_URL=your_api_endpoint
NEXT_PUBLIC_MODEL_ENDPOINT=your_model_endpoint
```

## API Integration

To connect with your backend:

1. **User Authentication**: Implement proper authentication flow
2. **Data Endpoints**: Create APIs for user data, model predictions, and analytics
3. **Real-time Updates**: Consider WebSocket integration for live updates

## Performance Considerations

- **Data Caching**: Implement caching for frequently accessed data
- **Lazy Loading**: Components are already optimized with lazy loading
- **Image Optimization**: Use Next.js Image component for plot visualizations
- **Code Splitting**: Automatic with Next.js app directory structure

## Security

- Implement proper authentication and authorization
- Validate all user inputs
- Use HTTPS in production
- Sanitize data before display

## Contributing

1. Follow the existing code structure and naming conventions
2. Add TypeScript types for all new components
3. Include proper error handling
4. Test responsive design on multiple screen sizes

## License

This project is part of the Grab Credit Scoring System.