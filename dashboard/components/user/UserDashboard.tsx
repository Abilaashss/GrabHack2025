'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  ArrowLeftOnRectangleIcon, 
  ChartBarIcon,
  CreditCardIcon,
  UserIcon,
  Cog6ToothIcon,
  SparklesIcon,
  HomeIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'
import ThemeToggle from '@/components/ThemeToggle'
import PremiumChat from '@/components/chat/PremiumChat'
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
  const [isChatOpen, setIsChatOpen] = useState(false)

  const tabs = [
    { id: 'overview', name: 'Overview', icon: ChartBarIcon },
    { id: 'parameters', name: 'Parameters', icon: CreditCardIcon },
    { id: 'models', name: 'Models', icon: Cog6ToothIcon },
    { id: 'performance', name: 'Performance', icon: UserIcon },
  ]

  return (
    <div className="min-h-screen transition-all duration-500">
      {/* Premium Header */}
      <header className="sticky top-0 z-50 backdrop-blur-premium border-b border-slate-200/50 dark:border-slate-700/50">
        <div className="premium-card border-0 rounded-none shadow-sm">
          <div className="max-w-7xl mx-auto px-6 lg:px-8">
            <div className="flex justify-between items-center h-20">
              <div className="flex items-center space-x-6">
                <Link 
                  href="/" 
                  className="group flex items-center space-x-3 text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 transition-all duration-300"
                >
                  <div className="p-2 rounded-xl bg-primary-50 dark:bg-primary-900/30 group-hover:bg-primary-100 dark:group-hover:bg-primary-900/50 transition-all duration-300">
                    <HomeIcon className="w-5 h-5" />
                  </div>
                </Link>
                <div className="border-l border-slate-200 dark:border-slate-700 pl-6">
                  <h1 className="text-2xl font-bold gradient-text text-shadow-premium">
                    {user.partner_type} Portal
                  </h1>
                  <p className="text-sm text-slate-600 dark:text-slate-400 font-medium">
                    AI-Powered Credit Intelligence Dashboard
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-3 px-4 py-2 bg-gradient-to-r from-primary-50 to-emerald-50 dark:from-primary-900/30 dark:to-emerald-900/30 rounded-xl border border-primary-200/50 dark:border-primary-700/50">
                  <div className="w-2 h-2 bg-emerald-500 rounded-full pulse-glow"></div>
                  <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                    {user.partner_type} #{user.partner_id}
                  </span>
                </div>
                <ThemeToggle />
                <button
                  onClick={onLogout}
                  className="flex items-center space-x-2 px-4 py-2 text-sm text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-all duration-300"
                >
                  <ArrowLeftOnRectangleIcon className="w-4 h-4" />
                  <span>Logout</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Premium Navigation */}
      <div className="sticky top-20 z-40 backdrop-blur-premium border-b border-slate-200/30 dark:border-slate-700/30">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <nav className="flex space-x-2 py-4">
            {tabs.map((tab) => {
              const Icon = tab.icon
              const isActive = activeTab === tab.id
              return (
                <motion.button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`group relative flex items-center space-x-3 px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-300 ${
                    isActive
                      ? 'bg-gradient-to-r from-primary-500 to-emerald-500 text-white shadow-lg'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-white/50 dark:hover:bg-slate-800/50'
                  }`}
                >
                  <div className={`p-1 rounded-lg transition-all duration-300 ${
                    isActive 
                      ? 'bg-white/20' 
                      : 'bg-slate-100 dark:bg-slate-700 group-hover:bg-slate-200 dark:group-hover:bg-slate-600'
                  }`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <span>{tab.name}</span>
                  {isActive && (
                    <motion.div
                      layoutId="activeUserTab"
                      className="absolute inset-0 bg-gradient-to-r from-primary-500 to-emerald-500 rounded-xl -z-10"
                      transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                </motion.button>
              )
            })}
          </nav>
        </div>
      </div>

      {/* Premium Main Content */}
      <main className="max-w-7xl mx-auto px-6 lg:px-8 py-12 min-h-screen">
        <div className="relative">
          {/* Background decoration */}
          <div className="absolute inset-0 -z-10">
            <div className="absolute top-0 left-1/4 w-72 h-72 bg-primary-500/5 rounded-full blur-3xl"></div>
            <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-emerald-500/5 rounded-full blur-3xl"></div>
          </div>
          
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab === 'overview' && (
              <div className="space-y-8">
                <CreditScoreCard user={user} selectedModel={selectedModel} />
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
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
        </div>
      </main>

      {/* Premium AI Chat */}
      <PremiumChat
        userRole="user"
        userType={user.partner_type}
        userId={user.partner_id.toString()}
        isOpen={isChatOpen}
        onToggle={() => setIsChatOpen(!isChatOpen)}
      />
    </div>
  )
}