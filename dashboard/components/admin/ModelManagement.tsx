'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  CpuChipIcon,
  ChartBarIcon,
  PhotoIcon,
  EyeIcon,
  TruckIcon,
  BuildingStorefrontIcon
} from '@heroicons/react/24/outline'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { getModelPerformance, getAvailableModels } from '@/lib/modelService'

export default function ModelManagement() {
  // All hooks must be at the top
  const [modelPerformance, setModelPerformance] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [selectedUserType, setSelectedUserType] = useState<'drivers' | 'merchants'>('drivers')
  const [selectedVisualization, setSelectedVisualization] = useState<string | null>(null)
  const [selectedVisualizationModel, setSelectedVisualizationModel] = useState<string>('')

  useEffect(() => {
    loadModelData()
    
    // Add keyboard event listener for ESC key
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && selectedVisualization) {
        setSelectedVisualization(null)
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [selectedVisualization])

  const loadModelData = async () => {
    try {
      const performance = await getModelPerformance()
      setModelPerformance(performance)
    } catch (error) {
      console.error('Error loading model data:', error)
    } finally {
      setLoading(false)
    }
  }

  const availableModels = getAvailableModels()
  const currentPerformance = modelPerformance?.[selectedUserType] || []

  // Real visualization data from results folder
  const availableVisualizations = [
    { 
      name: 'Feature Importance', 
      models: ['random_forest', 'gradient_boosting'],
      description: 'Shows which features contribute most to predictions',
      files: ['feature_importance.png']
    },
    { 
      name: 'Prediction vs Actual', 
      models: availableModels.map(m => m.id),
      description: 'Scatter plot comparing predicted vs actual credit scores',
      files: ['prediction_vs_actual.png']
    },
    { 
      name: 'Score Distribution', 
      models: availableModels.map(m => m.id),
      description: 'Distribution of predicted credit scores',
      files: ['score_distribution.png']
    },
    { 
      name: 'Model Comparison', 
      models: ['all'],
      description: 'Performance comparison across all models',
      files: ['model_performance_comparison.png']
    },
  ]

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 dark:border-primary-400 mx-auto"></div>
        <p className="text-slate-600 dark:text-slate-400 mt-4">Loading model data...</p>
      </div>
    )
  }

  const handleVisualizationView = (vizName: string, modelName?: string) => {
    const userType = selectedUserType
    let fileName = ''
    
    if (vizName === 'Model Comparison') {
      fileName = 'model_performance_comparison.png'
    } else if (vizName === 'Feature Importance') {
      fileName = `${userType}_${modelName}_feature_importance.png`
    } else if (vizName === 'Prediction vs Actual') {
      fileName = `${userType}_${modelName}_prediction_vs_actual.png`
    } else if (vizName === 'Score Distribution') {
      fileName = `${userType}_${modelName}_score_distribution.png`
    }
    
    setSelectedVisualization(`/api/visualizations/${fileName}`)
    setSelectedVisualizationModel(modelName || 'all')
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* Model Overview */}
      <div className="premium-card rounded-3xl p-10 hover-lift">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-gradient-to-br from-primary-500 to-primary-600 rounded-2xl shadow-lg">
              <CpuChipIcon className="w-7 h-7 text-white" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Model Performance Hub</h3>
              <p className="text-slate-600 dark:text-slate-400">Advanced ML model monitoring and analysis</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-emerald-500 rounded-full pulse-glow"></div>
            <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Live Monitoring</span>
          </div>
        </div>

        <div className="flex space-x-2 mb-8 p-2 bg-slate-100/50 dark:bg-slate-800/50 rounded-2xl backdrop-blur-sm">
          <button
            onClick={() => setSelectedUserType('drivers')}
            className={`flex-1 flex items-center justify-center space-x-3 px-8 py-4 rounded-xl font-semibold transition-all duration-300 ${
              selectedUserType === 'drivers'
                ? 'bg-gradient-to-r from-accent-500 to-accent-600 text-white shadow-lg hover:shadow-xl'
                : 'text-slate-700 dark:text-slate-300 hover:bg-white dark:hover:bg-slate-700 hover:shadow-md'
            }`}
          >
            <TruckIcon className="w-5 h-5" />
            <span>Driver Models</span>
            {selectedUserType === 'drivers' && (
              <div className="w-2 h-2 bg-white rounded-full"></div>
            )}
          </button>
          <button
            onClick={() => setSelectedUserType('merchants')}
            className={`flex-1 flex items-center justify-center space-x-3 px-8 py-4 rounded-xl font-semibold transition-all duration-300 ${
              selectedUserType === 'merchants'
                ? 'bg-gradient-to-r from-primary-500 to-primary-600 text-white shadow-lg hover:shadow-xl'
                : 'text-slate-700 dark:text-slate-300 hover:bg-white dark:hover:bg-slate-700 hover:shadow-md'
            }`}
          >
            <BuildingStorefrontIcon className="w-5 h-5" />
            <span>Merchant Models</span>
            {selectedUserType === 'merchants' && (
              <div className="w-2 h-2 bg-white rounded-full"></div>
            )}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div className="text-center p-4 bg-primary-50 dark:bg-primary-900/20 rounded-xl">
            <div className="text-2xl font-bold text-primary-600 dark:text-primary-400">
              {availableModels.length}
            </div>
            <div className="text-sm text-primary-700 dark:text-primary-300">Active Models</div>
          </div>
          <div className="text-center p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-xl">
            <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
              {currentPerformance.length > 0 
                ? (currentPerformance.reduce((sum: number, m: any) => sum + m.accuracy, 0) / currentPerformance.length * 100).toFixed(1)
                : '0'
              }%
            </div>
            <div className="text-sm text-emerald-700 dark:text-emerald-300">Avg Accuracy</div>
          </div>
          <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-xl">
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              {currentPerformance.length > 0 
                ? Math.max(...currentPerformance.map((m: any) => m.r2)).toFixed(3)
                : '0.000'
              }
            </div>
            <div className="text-sm text-purple-700 dark:text-purple-300">Best R² Score</div>
          </div>
          <div className="text-center p-4 bg-accent-50 dark:bg-accent-900/20 rounded-xl">
            <div className="text-2xl font-bold text-accent-600 dark:text-accent-400">
              {currentPerformance.length > 0 
                ? Math.min(...currentPerformance.map((m: any) => m.mse)).toFixed(0)
                : '0'
              }
            </div>
            <div className="text-sm text-accent-700 dark:text-accent-300">Best MSE</div>
          </div>
        </div>

        {/* Performance Chart */}
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={currentPerformance}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="accuracy" fill="#10b981" name="Accuracy" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

        {/* Model Details */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="premium-card rounded-xl p-6">
            <h4 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Model Performance Details</h4>
            
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {currentPerformance.map((model: any, index: number) => (
                <motion.div
                  key={model.name}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className="p-4 border border-slate-200 dark:border-slate-600 rounded-lg hover:border-primary-500 dark:hover:border-primary-400 transition-colors bg-white dark:bg-slate-800"
                >
                  <div className="flex items-center justify-between mb-3">
                    <h5 className="font-medium text-slate-900 dark:text-slate-100">{model.name}</h5>
                    <div className="text-sm font-semibold text-emerald-600 dark:text-emerald-400">
                      {(model.accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="bg-slate-50 dark:bg-slate-700 p-2 rounded-lg">
                      <span className="text-slate-600 dark:text-slate-300 block text-xs">R² Score</span>
                      <span className="font-bold text-primary-600 dark:text-primary-400 text-lg">
                        {model.r2.toFixed(3)}
                      </span>
                    </div>
                    <div className="bg-slate-50 dark:bg-slate-700 p-2 rounded-lg">
                      <span className="text-slate-600 dark:text-slate-300 block text-xs">MSE</span>
                      <span className="font-medium text-slate-900 dark:text-slate-100">
                        {model.mse.toFixed(0)}
                      </span>
                    </div>
                    <div className="bg-slate-50 dark:bg-slate-700 p-2 rounded-lg">
                      <span className="text-slate-600 dark:text-slate-300 block text-xs">Accuracy</span>
                      <span className="font-medium text-emerald-600 dark:text-emerald-400">
                        {(model.accuracy * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="bg-slate-50 dark:bg-slate-700 p-2 rounded-lg">
                      <span className="text-slate-600 dark:text-slate-300 block text-xs">MAE</span>
                      <span className="font-medium text-slate-900 dark:text-slate-100">
                        {model.mae?.toFixed(2) || 'N/A'}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>        {/* Available Visualizations */}
        <div className="premium-card rounded-xl p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg shadow-lg">
              <PhotoIcon className="w-5 h-5 text-white" />
            </div>
            <div>
              <h4 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Model Visualizations</h4>
              <p className="text-sm text-slate-600 dark:text-slate-400">Interactive charts and performance plots</p>
            </div>
          </div>
          
          <div className="space-y-4">
            {availableVisualizations.map((viz, index) => (
              <motion.div
                key={viz.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                className="p-4 border border-slate-200 dark:border-slate-600 rounded-xl hover:border-primary-500 dark:hover:border-primary-400 transition-all duration-200 bg-white dark:bg-slate-800 hover:shadow-md"
              >
                <div className="flex items-center justify-between mb-3">
                  <h5 className="font-medium text-slate-900 dark:text-slate-100">{viz.name}</h5>
                  {viz.models.includes('all') ? (
                    <button 
                      onClick={() => handleVisualizationView(viz.name)}
                      className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 bg-primary-50 hover:bg-primary-100 dark:bg-primary-900/30 dark:hover:bg-primary-900/50 rounded-lg transition-all duration-200"
                      title={`View ${viz.name}`}
                    >
                      <EyeIcon className="w-4 h-4" />
                      <span>View</span>
                    </button>
                  ) : (
                    <div className="flex flex-wrap gap-2">
                      {viz.models.slice(0, 3).map(model => (
                        <button 
                          key={model}
                          onClick={() => handleVisualizationView(viz.name, model)}
                          className="flex items-center space-x-1 px-2 py-1 text-xs font-medium bg-slate-100 hover:bg-primary-100 dark:bg-slate-700 dark:hover:bg-primary-800 text-slate-600 hover:text-primary-600 dark:text-slate-300 dark:hover:text-primary-300 rounded-md transition-all duration-200"
                          title={`View ${viz.name} for ${model}`}
                        >
                          <EyeIcon className="w-3 h-3" />
                          <span>{model.replace('_', ' ')}</span>
                        </button>
                      ))}
                      {viz.models.length > 3 && (
                        <span className="text-xs text-slate-500 dark:text-slate-400 px-2 py-1">
                          +{viz.models.length - 3} more
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">{viz.description}</p>
                <div className="flex items-center space-x-2 text-xs text-slate-500 dark:text-slate-400">
                  <span className="inline-flex items-center px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded-md">
                    {viz.models.includes('all') ? 'All models' : `${viz.models.length} models`}
                  </span>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="mt-6 p-4 bg-gradient-to-r from-primary-50 to-purple-50 dark:from-primary-900/20 dark:to-purple-900/20 rounded-xl border border-primary-200 dark:border-primary-700">
            <div className="flex items-start space-x-3">
              <PhotoIcon className="w-5 h-5 text-primary-600 dark:text-primary-400 mt-0.5" />
              <div>
                <h5 className="font-medium text-primary-900 dark:text-primary-100 mb-1">Visualization Files</h5>
                <p className="text-sm text-primary-700 dark:text-primary-300">
                  All model visualizations are stored in the <code className="bg-primary-100 dark:bg-primary-800 px-1 py-0.5 rounded text-xs">results/plots</code> directory. 
                  Click the view buttons to display individual charts and performance metrics.
                </p>
              </div>
            </div>
          </div>

          {/* Visualization Modal */}
          {selectedVisualization && (
            <div 
              className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-[60] p-4"
              onClick={(e) => {
                // Close modal when clicking outside
                if (e.target === e.currentTarget) {
                  setSelectedVisualization(null)
                }
              }}
            >
              <motion.div 
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="bg-white dark:bg-slate-800 rounded-2xl max-w-5xl max-h-[90vh] overflow-auto shadow-2xl border border-slate-200 dark:border-slate-700"
                onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside modal
              >
                <div className="sticky top-0 bg-white dark:bg-slate-800 p-6 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between z-10">
                  <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center space-x-3">
                    <PhotoIcon className="w-6 h-6 text-primary-500" />
                    <span>Model Visualization - {selectedVisualizationModel}</span>
                  </h3>
                  <button
                    onClick={() => setSelectedVisualization(null)}
                    className="flex items-center justify-center w-10 h-10 text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-full transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500"
                    title="Close visualization"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <div className="p-6">
                  <div className="text-center">
                    <img 
                      src={selectedVisualization} 
                      alt={`Model Visualization - ${selectedVisualizationModel}`}
                      className="max-w-full h-auto rounded-lg shadow-md border border-slate-200 dark:border-slate-600"
                      onError={() => {
                        console.error('Failed to load visualization:', selectedVisualization)
                        setSelectedVisualization(null)
                      }}
                    />
                    <p className="mt-4 text-sm text-slate-600 dark:text-slate-400">
                      Click the X button above or press ESC to close this visualization.
                    </p>
                  </div>
                </div>
              </motion.div>
            </div>
          )}
        </div>
      </div>

      {/* Model Configuration */}
      <div className="premium-card rounded-xl p-8">
        <div className="flex items-center space-x-4 mb-8">
          <div className="p-3 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-2xl shadow-lg">
            <ChartBarIcon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h4 className="text-xl font-bold text-slate-900 dark:text-slate-100">Model Configuration</h4>
            <p className="text-slate-600 dark:text-slate-400">Active machine learning models</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {availableModels.map((model, index) => (
            <motion.div
              key={model.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className="p-6 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 rounded-xl hover:border-primary-500 dark:hover:border-primary-400 transition-all duration-200 hover:shadow-lg group"
            >
              <div className="text-center">
                <div className="w-14 h-14 bg-gradient-to-br from-primary-100 to-primary-200 dark:from-primary-800 dark:to-primary-700 rounded-xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-200">
                  <CpuChipIcon className="w-7 h-7 text-primary-600 dark:text-primary-400" />
                </div>
                <h5 className="font-semibold text-slate-900 dark:text-slate-100 mb-2">{model.name}</h5>
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">{model.type}</p>
                <div className="inline-flex items-center space-x-2 px-3 py-1 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-full text-xs font-medium">
                  <div className="w-2 h-2 bg-emerald-500 rounded-full pulse-glow"></div>
                  <span>Active</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  )
}