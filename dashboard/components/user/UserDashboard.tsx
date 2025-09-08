'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  ArrowLeftOnRectangleIcon, 
  ChartBarIcon,
  CreditCardIcon,
  UserIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline'
import CreditScoreCard from './CreditScoreCard'
import ParametersView from './ParametersView'
import ModelSelector from './ModelSelector'
import PerformanceMetrics from './PerformanceMetrics'

interface UserDashboardProps {
  user: any
  onLogout: () => void
}

export default function UserDashboard({ user, onLogout }: UserDashboardProps) {
  const [selectedModel, setSelectedModel] = useState('random_forest')
  const [activeTab, setActiveTab] = useState('overview')

  const tabs = [
    { id: 'overview', name: 'Overview', icon: ChartBarIcon },
    { id: 'parameters', name: 'Parameters', icon: CreditCardIcon },
    { id: 'models', name: 'Models', icon: Cog6ToothIcon },
    { id: 'performance', name: 'Performance', icon: UserIcon },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-semibold text-gray-900">
                Credit Score Dashboard
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                Welcome, {user.partner_type} #{user.partner_id}
              </div>
              <button
                onClick={onLogout}
                className="flex items-center space-x-2 px-4 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                <ArrowLeftOnRectangleIcon className="w-4 h-4" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-grab-500 text-grab-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{tab.name}</span>
                </button>
              )
            })}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <CreditScoreCard user={user} selectedModel={selectedModel} />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ParametersView user={user} />
                <PerformanceMetrics user={user} />
              </div>
            </div>
          )}

          {activeTab === 'parameters' && (
            <ParametersView user={user} detailed={true} />
          )}

          {activeTab === 'models' && (
            <ModelSelector 
              user={user}
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
            />
          )}

          {activeTab === 'performance' && (
            <PerformanceMetrics user={user} detailed={true} />
          )}
        </motion.div>
      </main>
    </div>
  )
}