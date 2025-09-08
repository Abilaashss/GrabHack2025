'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  ChartBarIcon,
  UsersIcon,
  CpuChipIcon,
  MagnifyingGlassIcon,
  HomeIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'
import ThemeToggle from '@/components/ThemeToggle'
import AnalyticsDashboard from './AnalyticsDashboard'
import UserSearch from './UserSearch'
import ModelManagement from './ModelManagement'
import SystemOverview from './SystemOverview'

export default function AdminDashboard() {
  const [activeTab, setActiveTab] = useState('overview')

  const tabs = [
    { id: 'overview', name: 'Overview', icon: HomeIcon },
    { id: 'analytics', name: 'Analytics', icon: ChartBarIcon },
    { id: 'users', name: 'User Search', icon: MagnifyingGlassIcon },
    { id: 'models', name: 'Models', icon: CpuChipIcon },
  ]

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 transition-colors">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 shadow-sm border-b border-slate-200 dark:border-slate-700 transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <Link href="/" className="text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 transition-colors">
                <HomeIcon className="w-6 h-6" />
              </Link>
              <div>
                <h1 className="text-xl font-bold text-slate-900 dark:text-slate-100">Grab Credit Analytics</h1>
                <p className="text-xs text-slate-500 dark:text-slate-400">Administrator Dashboard</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <ThemeToggle />
              <div className="px-3 py-1 bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300 rounded-full text-sm font-medium">
                Admin Panel
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-1">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-6 font-medium text-sm transition-all relative ${
                    activeTab === tab.id
                      ? 'text-primary-600 dark:text-primary-400 bg-primary-50 dark:bg-primary-900/20'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{tab.name}</span>
                  {activeTab === tab.id && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-500 dark:bg-primary-400"></div>
                  )}
                </button>
              )
            })}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 bg-slate-50 dark:bg-slate-900 min-h-screen transition-colors">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {activeTab === 'overview' && <SystemOverview />}
          {activeTab === 'analytics' && <AnalyticsDashboard />}
          {activeTab === 'users' && <UserSearch />}
          {activeTab === 'models' && <ModelManagement />}
        </motion.div>
      </main>
    </div>
  )
}