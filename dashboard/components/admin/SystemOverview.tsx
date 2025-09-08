'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  UsersIcon,
  TruckIcon,
  BuildingStorefrontIcon,
  ChartBarIcon,
  CpuChipIcon,
  ClockIcon
} from '@heroicons/react/24/outline'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts'
import { getSystemStats } from '@/lib/dataService'

export default function SystemOverview() {
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadSystemStats()
  }, [])

  const loadSystemStats = async () => {
    try {
      const systemStats = await getSystemStats()
      setStats(systemStats)
    } catch (error) {
      console.error('Error loading system stats:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto"></div>
        <p className="text-gray-600 mt-4">Loading system overview...</p>
      </div>
    )
  }

  const COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444']

  const overviewCards = [
    {
      title: 'Total Users',
      value: stats?.totalUsers || 0,
      icon: UsersIcon,
      color: 'bg-blue-500',
      change: '+12%',
    },
    {
      title: 'Active Drivers',
      value: stats?.totalDrivers || 0,
      icon: TruckIcon,
      color: 'bg-green-500',
      change: '+8%',
    },
    {
      title: 'Active Merchants',
      value: stats?.totalMerchants || 0,
      icon: BuildingStorefrontIcon,
      color: 'bg-purple-500',
      change: '+15%',
    },
    {
      title: 'Avg Credit Score',
      value: stats?.avgCreditScore || 0,
      icon: ChartBarIcon,
      color: 'bg-yellow-500',
      change: '+2.3%',
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {overviewCards.map((card, index) => {
          const Icon = card.icon
          return (
            <motion.div
              key={card.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="bg-white rounded-xl p-6 card-shadow-lg"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{card.title}</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {typeof card.value === 'number' && card.title.includes('Score') 
                      ? card.value.toFixed(0)
                      : card.value.toLocaleString()
                    }
                  </p>
                  <p className="text-sm text-green-600 font-medium">{card.change}</p>
                </div>
                <div className={`p-3 rounded-lg ${card.color}`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Credit Score Distribution */}
        <div className="bg-white rounded-xl p-6 card-shadow-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Credit Score Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={stats?.scoreDistribution || []}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {(stats?.scoreDistribution || []).map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* User Type Distribution */}
        <div className="bg-white rounded-xl p-6 card-shadow-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">User Type Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats?.userTypeDistribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#22c55e" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Model Performance Summary */}
      <div className="bg-white rounded-xl p-6 card-shadow-lg">
        <div className="flex items-center space-x-3 mb-6">
          <CpuChipIcon className="w-6 h-6 text-primary-500" />
          <h3 className="text-lg font-semibold text-gray-900">Model Performance Summary</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">8</div>
            <div className="text-sm text-gray-600">Active Models</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">98.5%</div>
            <div className="text-sm text-gray-600">Avg Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">1.2s</div>
            <div className="text-sm text-gray-600">Avg Response Time</div>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-xl p-6 card-shadow-lg">
        <div className="flex items-center space-x-3 mb-6">
          <ClockIcon className="w-6 h-6 text-primary-500" />
          <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
        </div>

        <div className="space-y-4">
          {[
            { action: 'New driver registered', user: 'Driver #12345', time: '2 minutes ago' },
            { action: 'Credit score updated', user: 'Merchant #67890', time: '5 minutes ago' },
            { action: 'Model retrained', user: 'System', time: '1 hour ago' },
            { action: 'New merchant registered', user: 'Merchant #11111', time: '2 hours ago' },
          ].map((activity, index) => (
            <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <div className="font-medium text-gray-900">{activity.action}</div>
                <div className="text-sm text-gray-600">{activity.user}</div>
              </div>
              <div className="text-sm text-gray-500">{activity.time}</div>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  )
}