'use client'

import { motion } from 'framer-motion'
import { TrendingUpIcon, TrendingDownIcon } from '@heroicons/react/24/outline'

interface CreditScoreCardProps {
  user: any
  selectedModel: string
}

export default function CreditScoreCard({ user, selectedModel }: CreditScoreCardProps) {
  const creditScore = user.credit_score
  
  // Calculate score category and color
  const getScoreCategory = (score: number) => {
    if (score >= 700) return { category: 'Excellent', color: 'text-green-600', bgColor: 'bg-green-100' }
    if (score >= 600) return { category: 'Good', color: 'text-blue-600', bgColor: 'bg-blue-100' }
    if (score >= 500) return { category: 'Fair', color: 'text-yellow-600', bgColor: 'bg-yellow-100' }
    return { category: 'Poor', color: 'text-red-600', bgColor: 'bg-red-100' }
  }

  const scoreInfo = getScoreCategory(creditScore)
  
  // Calculate percentage for circular progress
  const percentage = (creditScore / 850) * 100
  const circumference = 2 * Math.PI * 45
  const strokeDasharray = circumference
  const strokeDashoffset = circumference - (percentage / 100) * circumference

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="bg-white rounded-2xl p-8 card-shadow-lg"
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Your Credit Score</h2>
          <p className="text-gray-600">Based on {selectedModel.replace('_', ' ').toUpperCase()} model</p>
        </div>
        <div className={`px-3 py-1 rounded-full text-sm font-medium ${scoreInfo.bgColor} ${scoreInfo.color}`}>
          {scoreInfo.category}
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="relative w-32 h-32 mx-auto">
            <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 100 100">
              {/* Background circle */}
              <circle
                cx="50"
                cy="50"
                r="45"
                stroke="currentColor"
                strokeWidth="8"
                fill="transparent"
                className="text-gray-200"
              />
              {/* Progress circle */}
              <circle
                cx="50"
                cy="50"
                r="45"
                stroke="currentColor"
                strokeWidth="8"
                fill="transparent"
                strokeDasharray={strokeDasharray}
                strokeDashoffset={strokeDashoffset}
                className="text-grab-500 transition-all duration-1000 ease-out"
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-3xl font-bold text-gray-900">{creditScore}</div>
                <div className="text-sm text-gray-500">/ 850</div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex-1 space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-600">Score Range</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-red-500 via-yellow-500 via-blue-500 to-green-500 h-2 rounded-full relative"
              >
                <div 
                  className="absolute top-0 w-3 h-3 bg-gray-900 rounded-full transform -translate-y-0.5"
                  style={{ left: `${percentage}%`, marginLeft: '-6px' }}
                />
              </div>
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>300</span>
              <span>850</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="text-center">
              <div className="text-lg font-semibold text-gray-900">
                {user.average_rating?.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600">Avg Rating</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold text-gray-900">
                {user.tenure_months}
              </div>
              <div className="text-sm text-gray-600">Months</div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 p-4 bg-grab-50 rounded-lg">
        <h3 className="font-medium text-grab-900 mb-2">Score Insights</h3>
        <p className="text-sm text-grab-700">
          {creditScore >= 700 && "Excellent! You have access to the best rates and terms."}
          {creditScore >= 600 && creditScore < 700 && "Good score! You qualify for competitive rates."}
          {creditScore >= 500 && creditScore < 600 && "Fair score. Consider improving key metrics for better rates."}
          {creditScore < 500 && "Focus on improving your performance metrics to increase your score."}
        </p>
      </div>
    </motion.div>
  )
}