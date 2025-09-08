'use client'

import { motion } from 'framer-motion'
import { 
  ClockIcon, 
  StarIcon, 
  CurrencyDollarIcon,
  TruckIcon,
  UserGroupIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline'

interface ParametersViewProps {
  user: any
  detailed?: boolean
}

export default function ParametersView({ user, detailed = false }: ParametersViewProps) {
  const isDriver = user.partner_type === 'Driver'
  
  const getParameterIcon = (key: string) => {
    const iconMap: { [key: string]: any } = {
      tenure_months: ClockIcon,
      average_rating: StarIcon,
      monthly_earnings: CurrencyDollarIcon,
      monthly_sales: CurrencyDollarIcon,
      completion_rate: ChartBarIcon,
      acceptance_rate: ChartBarIcon,
      total_trips: TruckIcon,
      customer_complaints: UserGroupIcon,
      customer_compliments: UserGroupIcon,
    }
    return iconMap[key] || ChartBarIcon
  }

  const formatValue = (key: string, value: any) => {
    if (key.includes('rate') || key.includes('ratio')) {
      return `${(value * 100).toFixed(1)}%`
    }
    if (key.includes('earnings') || key.includes('sales') || key.includes('value')) {
      return `$${value.toLocaleString()}`
    }
    if (key.includes('rating')) {
      return `${value.toFixed(1)}/5.0`
    }
    if (typeof value === 'number' && value % 1 !== 0) {
      return value.toFixed(2)
    }
    return value
  }

  const getParameterColor = (key: string, value: any) => {
    if (key.includes('complaints') || key.includes('cancel') || key.includes('error')) {
      return value > 0.1 ? 'text-red-600' : 'text-green-600'
    }
    if (key.includes('rating') || key.includes('compliments') || key.includes('completion')) {
      return value > 4 || value > 0.8 ? 'text-green-600' : value > 3 || value > 0.6 ? 'text-yellow-600' : 'text-red-600'
    }
    return 'text-gray-900'
  }

  // Define key parameters to display
  const keyParameters = isDriver ? [
    'tenure_months',
    'average_rating', 
    'monthly_earnings',
    'completion_rate',
    'acceptance_rate',
    'total_trips',
    'customer_complaints',
    'customer_compliments'
  ] : [
    'tenure_months',
    'average_rating',
    'monthly_sales', 
    'order_acceptance_rate',
    'avg_order_value',
    'repeat_customer_rate',
    'customer_complaints',
    'customer_compliments'
  ]

  const allParameters = Object.keys(user).filter(key => 
    !['partner_id', 'partner_type', 'credit_score'].includes(key)
  )

  const parametersToShow = detailed ? allParameters : keyParameters

  const getParameterLabel = (key: string) => {
    const labelMap: { [key: string]: string } = {
      tenure_months: 'Tenure (Months)',
      average_rating: 'Average Rating',
      monthly_earnings: 'Monthly Earnings',
      monthly_sales: 'Monthly Sales',
      completion_rate: 'Completion Rate',
      acceptance_rate: 'Acceptance Rate',
      order_acceptance_rate: 'Order Acceptance Rate',
      total_trips: 'Total Trips',
      avg_order_value: 'Average Order Value',
      repeat_customer_rate: 'Repeat Customer Rate',
      customer_complaints: 'Customer Complaints',
      customer_compliments: 'Customer Compliments',
      earnings_stability: 'Earnings Stability',
      digital_payment_ratio: 'Digital Payment Ratio',
      age_group: 'Age Group',
      gender: 'Gender',
      ethnicity: 'Ethnicity',
      education_level: 'Education Level',
      location: 'Location',
      city_tier: 'City Tier',
    }
    return labelMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-white rounded-2xl p-6 card-shadow-lg"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">
          {detailed ? 'All Parameters' : 'Key Parameters'}
        </h3>
        <div className="text-sm text-gray-500">
          {user.partner_type} Profile
        </div>
      </div>

      <div className={`grid gap-4 ${detailed ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3' : 'grid-cols-1 md:grid-cols-2'}`}>
        {parametersToShow.map((key, index) => {
          const Icon = getParameterIcon(key)
          const value = user[key]
          const formattedValue = formatValue(key, value)
          const valueColor = getParameterColor(key, value)

          return (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors"
            >
              <div className="flex items-center space-x-3">
                <div className="flex-shrink-0">
                  <Icon className="w-5 h-5 text-grab-500" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {getParameterLabel(key)}
                  </p>
                  <p className={`text-lg font-semibold ${valueColor}`}>
                    {formattedValue}
                  </p>
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>

      {!detailed && (
        <div className="mt-6 text-center">
          <p className="text-sm text-gray-600">
            View the Parameters tab for complete details
          </p>
        </div>
      )}
    </motion.div>
  )
}