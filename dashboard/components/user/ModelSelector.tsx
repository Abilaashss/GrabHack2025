'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { CpuChipIcon, ChartBarIcon } from '@heroicons/react/24/outline'
import { predictCreditScore } from '@/lib/modelService'

interface ModelSelectorProps {
  user: any
  selectedModel: string
  onModelChange: (model: string) => void
}

export default function ModelSelector({ user, selectedModel, onModelChange }: ModelSelectorProps) {
  const [predictions, setPredictions] = useState<{ [key: string]: number }>({})
  const [loading, setLoading] = useState(false)

  const availableModels = [
    { id: 'random_forest', name: 'Random Forest', description: 'Ensemble method with high accuracy' },
    { id: 'gradient_boosting', name: 'Gradient Boosting', description: 'Sequential learning for complex patterns' },
    { id: 'linear_regression', name: 'Linear Regression', description: 'Simple and interpretable model' },
    { id: 'ridge', name: 'Ridge Regression', description: 'Regularized linear model' },
    { id: 'lasso', name: 'Lasso Regression', description: 'Feature selection with regularization' },
    { id: 'elastic_net', name: 'Elastic Net', description: 'Combines Ridge and Lasso' },
    { id: 'svr', name: 'Support Vector Regression', description: 'Non-linear pattern recognition' },
    { id: 'mlp_regressor', name: 'Neural Network', description: 'Deep learning approach' },
  ]

  useEffect(() => {
    loadPredictions()
  }, [user])

  const loadPredictions = async () => {
    setLoading(true)
    const newPredictions: { [key: string]: number } = {}
    
    for (const model of availableModels) {
      try {
        const prediction = await predictCreditScore(user, model.id)
        newPredictions[model.id] = prediction
      } catch (error) {
        console.error(`Error predicting with ${model.id}:`, error)
        newPredictions[model.id] = user.credit_score // Fallback to actual score
      }
    }
    
    setPredictions(newPredictions)
    setLoading(false)
  }

  const getScoreColor = (score: number) => {
    if (score >= 700) return 'text-green-600 bg-green-100'
    if (score >= 600) return 'text-blue-600 bg-blue-100'
    if (score >= 500) return 'text-yellow-600 bg-yellow-100'
    return 'text-red-600 bg-red-100'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      <div className="bg-white rounded-2xl p-6 card-shadow-lg">
        <div className="flex items-center space-x-3 mb-6">
          <CpuChipIcon className="w-6 h-6 text-grab-500" />
          <h3 className="text-xl font-semibold text-gray-900">Model Comparison</h3>
        </div>

        <p className="text-gray-600 mb-6">
          Compare your credit score predictions across different machine learning models. 
          Each model uses different algorithms and may provide varying insights.
        </p>

        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-grab-500 mx-auto"></div>
            <p className="text-gray-600 mt-4">Loading model predictions...</p>
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {availableModels.map((model, index) => {
              const prediction = predictions[model.id] || user.credit_score
              const isSelected = selectedModel === model.id
              const scoreColorClass = getScoreColor(prediction)

              return (
                <motion.button
                  key={model.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  onClick={() => onModelChange(model.id)}
                  className={`p-4 rounded-lg border-2 text-left transition-all hover:shadow-md ${
                    isSelected
                      ? 'border-grab-500 bg-grab-50'
                      : 'border-gray-200 bg-white hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-gray-900">{model.name}</h4>
                    <div className={`px-2 py-1 rounded text-sm font-semibold ${scoreColorClass}`}>
                      {Math.round(prediction)}
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">{model.description}</p>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">
                      Difference: {prediction > user.credit_score ? '+' : ''}{Math.round(prediction - user.credit_score)}
                    </span>
                    {isSelected && (
                      <div className="flex items-center space-x-1 text-grab-600">
                        <ChartBarIcon className="w-4 h-4" />
                        <span className="text-xs font-medium">Selected</span>
                      </div>
                    )}
                  </div>
                </motion.button>
              )
            })}
          </div>
        )}
      </div>

      <div className="bg-white rounded-2xl p-6 card-shadow-lg">
        <h4 className="text-lg font-semibold text-gray-900 mb-4">
          Selected Model: {availableModels.find(m => m.id === selectedModel)?.name}
        </h4>
        
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h5 className="font-medium text-gray-900 mb-2">Current Prediction</h5>
              <div className="text-2xl font-bold text-grab-600">
                {Math.round(predictions[selectedModel] || user.credit_score)}
              </div>
            </div>
            <div>
              <h5 className="font-medium text-gray-900 mb-2">Model Description</h5>
              <p className="text-sm text-gray-600">
                {availableModels.find(m => m.id === selectedModel)?.description}
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}