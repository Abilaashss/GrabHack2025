'use client'

import { motion } from 'framer-motion'
import { 
  ArrowTrendingUpIcon as TrendingUpIcon, 
  ArrowTrendingDownIcon as TrendingDownIcon,
  ChartBarIcon,
  ClockIcon
} from '@heroicons/react/24/outline'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts'

interface PerformanceMetricsProps {
  user: any
  detailed?: boolean
}

export default function PerformanceMetrics({ user, detailed = false }: PerformanceMetricsProps) {
  const isDriver = user.partner_type === 'Driver'

  // Generate mock historical data for demonstration
  const generateHistoricalData = () => {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    return months.map((month, index) => ({
      month,
      creditScore: user.credit_score + (Math.random() - 0.5) * 50,
      earnings: (user.monthly_earnings || user.monthly_sales) * (0.8 + Math.random() * 0.4),
      rating: user.average_rating + (Math.random() - 0.5) * 0.5,
    }))
  }

  const historicalData = generateHistoricalData()

  // Performance indicators
  const performanceMetrics = isDriver ? [
    {
      label: 'Completion Rate',
      value: user.completion_rate,
      target: 0.9,
      format: 'percentage',
      icon: ChartBarIcon,
    },
    {
      label: 'Acceptance Rate', 
      value: user.acceptance_rate,
      target: 0.8,
      format: 'percentage',
      icon: TrendingUpIcon,
    },
    {
      label: 'Cancel Rate',
      value: user.cancel_rate,
      target: 0.05,
      format: 'percentage',
      icon: TrendingDownIcon,
      inverse: true,
    },
    {
      label: 'Hours Online/Week',
      value: user.hours_online_per_week,
      target: 40,
      format: 'number',
      icon: ClockIcon,
    },
  ] : [
    {
      label: 'Order Acceptance Rate',
      value: user.order_acceptance_rate,
      target: 0.9,
      format: 'percentage',
      icon: ChartBarIcon,
    },
    {
      label: 'Repeat Customer Rate',
      value: user.repeat_customer_rate,
      target: 0.3,
      format: 'percentage',
      icon: TrendingUpIcon,
    },
    {
      label: 'Order Error Rate',
      value: user.order_error_rate,
      target: 0.02,
      format: 'percentage',
      icon: TrendingDownIcon,
      inverse: true,
    },
    {
      label: 'Avg Preparation Time',
      value: user.avg_preparation_time_mins,
      target: 20,
      format: 'minutes',
      icon: ClockIcon,
      inverse: true,
    },
  ]

  const formatMetricValue = (value: number, format: string) => {
    switch (format) {
      case 'percentage':
        return `${(value * 100).toFixed(1)}%`
      case 'minutes':
        return `${value.toFixed(0)} min`
      default:
        return value.toFixed(1)
    }
  }

  const getMetricStatus = (value: number, target: number, inverse = false) => {
    const ratio = value / target
    const isGood = inverse ? ratio <= 1 : ratio >= 1
    
    if (isGood) return { color: 'text-green-600', bg: 'bg-green-100', status: 'Good' }
    if (ratio > 0.8) return { color: 'text-yellow-600', bg: 'bg-yellow-100', status: 'Fair' }
    return { color: 'text-red-600', bg: 'bg-red-100', status: 'Needs Improvement' }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* Performance Metrics Cards */}
      <div className="bg-white rounded-2xl p-6 card-shadow-lg">
        <h3 className="text-xl font-semibold text-gray-900 mb-6">Performance Metrics</h3>
        
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {performanceMetrics.map((metric, index) => {
            const Icon = metric.icon
            const status = getMetricStatus(metric.value, metric.target, metric.inverse)
            
            return (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                className="bg-gray-50 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <Icon className="w-5 h-5 text-grab-500" />
                  <span className={`px-2 py-1 rounded text-xs font-medium ${status.bg} ${status.color}`}>
                    {status.status}
                  </span>
                </div>
                <div className="text-2xl font-bold text-gray-900 mb-1">
                  {formatMetricValue(metric.value, metric.format)}
                </div>
                <div className="text-sm text-gray-600">
                  {metric.label}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Target: {formatMetricValue(metric.target, metric.format)}
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* Historical Performance Chart */}
      {detailed && (
        <div className="bg-white rounded-2xl p-6 card-shadow-lg">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">Credit Score Trend</h3>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="creditScore" 
                  stroke="#22c55e" 
                  strokeWidth={2}
                  dot={{ fill: '#22c55e' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Earnings Performance */}
      {detailed && (
        <div className="bg-white rounded-2xl p-6 card-shadow-lg">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">
            {isDriver ? 'Earnings' : 'Sales'} Performance
          </h3>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, isDriver ? 'Earnings' : 'Sales']} />
                <Bar dataKey="earnings" fill="#22c55e" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Performance Summary */}
      <div className="bg-white rounded-2xl p-6 card-shadow-lg">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Performance Summary</h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
            <div>
              <h4 className="font-medium text-green-900">Strengths</h4>
              <p className="text-sm text-green-700">
                {user.average_rating >= 4.5 && "Excellent customer ratings"}
                {user.completion_rate >= 0.9 && " • High completion rate"}
                {(user.customer_compliments || 0) > (user.customer_complaints || 0) && " • More compliments than complaints"}
              </p>
            </div>
            <TrendingUpIcon className="w-8 h-8 text-green-600" />
          </div>

          <div className="flex items-center justify-between p-4 bg-yellow-50 rounded-lg">
            <div>
              <h4 className="font-medium text-yellow-900">Areas for Improvement</h4>
              <p className="text-sm text-yellow-700">
                {user.average_rating < 4.0 && "Focus on improving customer satisfaction"}
                {(user.cancel_rate || 0) > 0.1 && " • Reduce cancellation rate"}
                {(user.customer_complaints || 0) > 2 && " • Address customer complaints"}
              </p>
            </div>
            <ChartBarIcon className="w-8 h-8 text-yellow-600" />
          </div>
        </div>
      </div>
    </motion.div>
  )
}