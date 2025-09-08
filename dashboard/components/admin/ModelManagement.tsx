'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  CpuChipIcon,
  ChartBarIcon,
  PhotoIcon,
  EyeIcon
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
  }, [])

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
      <div className="premium-card rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-6">
          <CpuChipIcon className="w-6 h-6 text-primary-500 dark:text-primary-400" />
          <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100">Model Performance Overview</h3>
        </div>

        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setSelectedUserType('drivers')}
            className={`px-6 py-3 rounded-xl font-medium transition-all ${
              selectedUserType === 'drivers'
                ? 'bg-accent-500 dark:bg-accent-600 text-white shadow-sm'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
            }`}
          >
            Driver Models
          </button>
          <button
            onClick={() => setSelectedUserType('merchants')}
            className={`px-6 py-3 rounded-xl font-medium transition-all ${
              selectedUserType === 'merchants'
                ? 'bg-primary-500 dark:bg-primary-600 text-white shadow-sm'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
            }`}
          >
            Merchant Models
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div className="text-center p-4 bg-primary-50 rounded-xl">
            <div className="text-2xl font-bold text-primary-600">
              {availableModels.length}
            </div>
            <div className="text-sm text-primary-700">Active Models</div>
          </div>
          <div className="text-center p-4 bg-emerald-50 rounded-xl">
            <div className="text-2xl font-bold text-emerald-600">
              {currentPerformance.length > 0 
                ? (currentPerformance.reduce((sum: number, m: any) => sum + m.accuracy, 0) / currentPerformance.length * 100).toFixed(1)
                : '0'
              }%
            </div>
            <div className="text-sm text-emerald-700">Avg Accuracy</div>
          </div>
          <div className="text-center p-4 bg-accent-50 rounded-xl">
            <div className="text-2xl font-bold text-accent-600">
              {currentPerformance.length > 0 
                ? Math.min(...currentPerformance.map((m: any) => m.mse)).toFixed(0)
                : '0'
              }
            </div>
            <div className="text-sm text-accent-700">Best MSE</div>
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
        <div className="bg-white rounded-xl p-6 card-shadow-lg">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Model Performance Details</h4>
          
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {currentPerformance.map((model: any, index: number) => (
              <motion.div
                key={model.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                className="p-4 border border-gray-200 rounded-lg hover:border-primary-500 transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <h5 className="font-medium text-gray-900">{model.name}</h5>
                  <div className="text-sm font-semibold text-green-600">
                    {(model.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">MSE:</span>
                    <span className="ml-1 font-medium">{model.mse.toFixed(0)}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">R²:</span>
                    <span className="ml-1 font-medium">{model.r2.toFixed(3)}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Accuracy:</span>
                    <span className="ml-1 font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Available Visualizations */}
        <div className="bg-white rounded-xl p-6 card-shadow-lg">
          <div className="flex items-center space-x-3 mb-4">
            <PhotoIcon className="w-5 h-5 text-primary-500" />
            <h4 className="text-lg font-semibold text-gray-900">Model Visualizations</h4>
          </div>
          
          <div className="space-y-3">
            {availableVisualizations.map((viz, index) => (
              <motion.div
                key={viz.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                className="p-4 border border-slate-200 rounded-xl hover:border-primary-500 transition-colors bg-white"
              >
                <div className="flex items-center justify-between mb-2">
                  <h5 className="font-medium text-slate-900">{viz.name}</h5>
                  {viz.models.includes('all') ? (
                    <button 
                      onClick={() => handleVisualizationView(viz.name)}
                      className="p-2 text-slate-400 hover:text-primary-500 hover:bg-primary-50 rounded-lg transition-colors"
                    >
                      <EyeIcon className="w-4 h-4" />
                    </button>
                  ) : (
                    <div className="flex space-x-1">
                      {viz.models.slice(0, 3).map(model => (
                        <button 
                          key={model}
                          onClick={() => handleVisualizationView(viz.name, model)}
                          className="p-1 text-xs bg-slate-100 hover:bg-primary-100 text-slate-600 hover:text-primary-600 rounded transition-colors"
                          title={`View ${viz.name} for ${model}`}
                        >
                          <EyeIcon className="w-3 h-3" />
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <p className="text-sm text-slate-600 mb-2">{viz.description}</p>
                <div className="text-xs text-slate-500">
                  Available for: {viz.models.includes('all') ? 'All models' : `${viz.models.length} models`}
                </div>
              </motion.div>
            ))}
          </div>

          <div className="mt-6 p-4 bg-primary-50 rounded-xl">
            <h5 className="font-medium text-primary-900 mb-2">Visualization Files</h5>
            <p className="text-sm text-primary-700">
              All model visualizations are stored in the <code>results/plots</code> directory. 
              Click the eye icon to view individual charts and performance metrics.
            </p>
          </div>

          {/* Visualization Modal */}
          {selectedVisualization && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
              <div className="bg-white dark:bg-slate-800 rounded-2xl max-w-4xl max-h-[90vh] overflow-auto">
                <div className="p-6 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                    Model Visualization - {selectedVisualizationModel}
                  </h3>
                  <button
                    onClick={() => setSelectedVisualization(null)}
                    className="p-2 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <div className="p-6">
                  <img 
                    src={selectedVisualization} 
                    alt="Model Visualization"
                    className="w-full h-auto rounded-lg shadow-sm"
                    onError={() => {
                      console.error('Failed to load visualization')
                      setSelectedVisualization(null)
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Model Configuration */}
      <div className="bg-white rounded-xl p-6 card-shadow-lg">
        <div className="flex items-center space-x-3 mb-6">
          <ChartBarIcon className="w-6 h-6 text-primary-500" />
          <h4 className="text-lg font-semibold text-gray-900">Model Configuration</h4>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {availableModels.map((model, index) => (
            <motion.div
              key={model.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className="p-4 border border-gray-200 rounded-lg hover:border-primary-500 transition-colors"
            >
              <div className="text-center">
                <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                  <CpuChipIcon className="w-6 h-6 text-primary-600" />
                </div>
                <h5 className="font-medium text-gray-900 mb-1">{model.name}</h5>
                <p className="text-xs text-gray-600 mb-2">{model.type}</p>
                <div className="text-xs text-green-600 font-medium">Active</div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  )
}