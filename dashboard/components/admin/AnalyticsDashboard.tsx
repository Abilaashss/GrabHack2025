'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  ChartBarIcon,
  ArrowTrendingUpIcon as TrendingUpIcon,
  ArrowTrendingUpIcon,
  UsersIcon,
  CurrencyDollarIcon,
  StarIcon,
  BanknotesIcon,
  TruckIcon,
  BuildingStorefrontIcon
} from '@heroicons/react/24/outline'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  Area,
  AreaChart
} from 'recharts'
import { getAllUsers, loadData } from '@/lib/dataService'

export default function AnalyticsDashboard() {
  const [users, setUsers] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedView, setSelectedView] = useState<'overview' | 'drivers' | 'merchants'>('overview')

  useEffect(() => {
    loadAnalyticsData()
  }, [])

  const loadAnalyticsData = async () => {
    try {
      await loadData()
      const allUsers = await getAllUsers()
      setUsers(allUsers)
    } catch (error) {
      console.error('Error loading analytics data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-grab-500 mx-auto"></div>
          <p className="text-slate-600 mt-6 text-lg">Loading analytics data...</p>
        </div>
      </div>
    )
  }

  // Filter data based on selected view
  const getFilteredUsers = () => {
    if (selectedView === 'drivers') return users.filter(u => u.partner_type === 'Driver')
    if (selectedView === 'merchants') return users.filter(u => u.partner_type === 'Merchant')
    return users // overview shows all
  }

  const filteredUsers = getFilteredUsers()
  const drivers = users.filter(u => u.partner_type === 'Driver')
  const merchants = users.filter(u => u.partner_type === 'Merchant')

  // Credit Score Distribution with better ranges
  const scoreRanges = [
    { range: 'Poor (300-499)', count: 0, color: '#ef4444', drivers: 0, merchants: 0 },
    { range: 'Fair (500-599)', count: 0, color: '#f59e0b', drivers: 0, merchants: 0 },
    { range: 'Good (600-699)', count: 0, color: '#0ea5e9', drivers: 0, merchants: 0 },
    { range: 'Very Good (700-799)', count: 0, color: '#10b981', drivers: 0, merchants: 0 },
    { range: 'Excellent (800+)', count: 0, color: '#22c55e', drivers: 0, merchants: 0 },
  ]

  // For overview, show both drivers and merchants
  if (selectedView === 'overview') {
    scoreRanges.forEach(range => {
      range.drivers = 0
      range.merchants = 0
    })

    users.forEach(user => {
      const score = user.credit_score
      let rangeIndex = 0
      if (score >= 800) rangeIndex = 4
      else if (score >= 700) rangeIndex = 3
      else if (score >= 600) rangeIndex = 2
      else if (score >= 500) rangeIndex = 1
      else rangeIndex = 0
      
      if (user.partner_type === 'Driver') {
        scoreRanges[rangeIndex].drivers++
      } else {
        scoreRanges[rangeIndex].merchants++
      }
    })
  } else {
    // For specific views, show only count
    filteredUsers.forEach(user => {
      const score = user.credit_score
      let rangeIndex = 0
      if (score >= 800) rangeIndex = 4
      else if (score >= 700) rangeIndex = 3
      else if (score >= 600) rangeIndex = 2
      else if (score >= 500) rangeIndex = 1
      else rangeIndex = 0
      
      scoreRanges[rangeIndex].count++
    })
  }

  // Age Group Distribution
  const ageGroups = ['18-24', '25-34', '35-44', '45-54', '55+']
  const ageDistribution = ageGroups.map(age => {
    if (selectedView === 'overview') {
      return {
        age,
        drivers: drivers.filter(d => d.age_group === age).length,
        merchants: merchants.filter(m => m.age_group === age).length,
      }
    } else {
      return {
        age,
        count: filteredUsers.filter(u => u.age_group === age).length,
      }
    }
  })

  // Performance vs Credit Score correlation
  const performanceData = filteredUsers.slice(0, 500).map(user => ({
    creditScore: user.credit_score,
    rating: user.average_rating,
    earnings: user.monthly_earnings || user.monthly_sales,
    type: user.partner_type,
    tenure: user.tenure_months,
    digitalPayment: user.digital_payment_ratio,
  }))

  // Top Performers based on selected view
  const getTopPerformers = () => {
    if (selectedView === 'drivers') {
      return {
        drivers: drivers.sort((a, b) => b.credit_score - a.credit_score).slice(0, 10),
        merchants: []
      }
    } else if (selectedView === 'merchants') {
      return {
        drivers: [],
        merchants: merchants.sort((a, b) => b.credit_score - a.credit_score).slice(0, 10)
      }
    } else {
      return {
        drivers: drivers.sort((a, b) => b.credit_score - a.credit_score).slice(0, 8),
        merchants: merchants.sort((a, b) => b.credit_score - a.credit_score).slice(0, 8)
      }
    }
  }

  const { drivers: topDrivers, merchants: topMerchants } = getTopPerformers()

  // Calculate key metrics based on filtered data
  const avgCreditScore = filteredUsers.length > 0 ? filteredUsers.reduce((sum, u) => sum + u.credit_score, 0) / filteredUsers.length : 0
  const highPerformers = filteredUsers.filter(u => u.credit_score >= 700).length
  const totalRevenue = filteredUsers.reduce((sum, u) => sum + (u.monthly_earnings || u.monthly_sales || 0), 0)
  const avgRating = filteredUsers.length > 0 ? filteredUsers.reduce((sum, u) => sum + u.average_rating, 0) / filteredUsers.length : 0

  // Advanced analytics
  const riskDistribution = [
    { risk: 'Low Risk (700+)', count: filteredUsers.filter(u => u.credit_score >= 700).length, color: '#22c55e' },
    { risk: 'Medium Risk (600-699)', count: filteredUsers.filter(u => u.credit_score >= 600 && u.credit_score < 700).length, color: '#f59e0b' },
    { risk: 'High Risk (<600)', count: filteredUsers.filter(u => u.credit_score < 600).length, color: '#ef4444' },
  ]

  // Earnings vs Credit Score correlation
  const earningsCorrelation = filteredUsers
    .filter(u => (u.monthly_earnings || u.monthly_sales) > 0)
    .map(u => ({
      creditScore: u.credit_score,
      earnings: u.monthly_earnings || u.monthly_sales,
      type: u.partner_type
    }))
    .slice(0, 300)

  // Digital payment adoption
  const digitalPaymentData = ageGroups.map(age => ({
    age,
    adoption: filteredUsers
      .filter(u => u.age_group === age)
      .reduce((sum, u) => sum + (u.digital_payment_ratio || 0), 0) / 
      Math.max(filteredUsers.filter(u => u.age_group === age).length, 1)
  }))



  return (
    <div className="space-y-8">
      {/* Premium Header with View Selector */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-8 mb-12"
      >
        <div className="space-y-2">
          <h2 className="text-4xl font-bold gradient-text text-shadow-premium">
            Analytics Dashboard
          </h2>
          <p className="text-lg text-slate-600 dark:text-slate-400 font-medium">
            Real-time insights powered by advanced machine learning
          </p>
          <div className="flex items-center space-x-4 text-sm text-slate-500 dark:text-slate-500">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full pulse-glow"></div>
              <span>Live Data</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <span>{filteredUsers.length.toLocaleString()} Records</span>
            </div>
          </div>
        </div>
        
        <div className="premium-card p-2 border border-slate-200/50 dark:border-slate-700/50">
          <div className="flex space-x-1">
            {[
              { id: 'overview', label: 'Overview', icon: ChartBarIcon, color: 'from-slate-500 to-slate-600' },
              { id: 'drivers', label: 'Drivers', icon: TruckIcon, color: 'from-blue-500 to-blue-600' },
              { id: 'merchants', label: 'Merchants', icon: BuildingStorefrontIcon, color: 'from-emerald-500 to-emerald-600' }
            ].map(({ id, label, icon: Icon, color }) => (
              <motion.button
                key={id}
                onClick={() => setSelectedView(id as any)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className={`relative flex items-center space-x-3 px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-300 ${
                  selectedView === id
                    ? `bg-gradient-to-r ${color} text-white shadow-lg`
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800/50'
                }`}
              >
                <div className={`p-1 rounded-lg transition-all duration-300 ${
                  selectedView === id 
                    ? 'bg-white/20' 
                    : 'bg-slate-100 dark:bg-slate-700'
                }`}>
                  <Icon className="w-4 h-4" />
                </div>
                <span>{label}</span>
                {selectedView === id && (
                  <motion.div
                    layoutId="activeView"
                    className={`absolute inset-0 bg-gradient-to-r ${color} rounded-xl -z-10`}
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
              </motion.button>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Premium Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="group premium-card rounded-2xl p-8 hover-lift interactive-element"
        >
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                <p className="text-sm font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wide">
                  Average Credit Score
                </p>
              </div>
              <p className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                {avgCreditScore.toFixed(0)}
              </p>
              <div className="flex items-center space-x-2">
                <div className="flex items-center space-x-1 px-2 py-1 bg-emerald-100 dark:bg-emerald-900/30 rounded-full">
                  <ArrowTrendingUpIcon className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                  <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">+2.3%</span>
                </div>
                <span className="text-xs text-slate-500 dark:text-slate-500">vs last month</span>
              </div>
            </div>
            <div className="p-4 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-2xl shadow-lg group-hover:shadow-xl transition-all duration-300">
              <ChartBarIcon className="w-7 h-7 text-white" />
            </div>
          </div>
          <div className="h-1 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full"></div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="group premium-card rounded-2xl p-8 hover-lift interactive-element"
        >
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-2 h-2 bg-primary-500 rounded-full"></div>
                <p className="text-sm font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wide">
                  High Performers
                </p>
              </div>
              <p className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                {highPerformers.toLocaleString()}
              </p>
              <div className="flex items-center space-x-2">
                <div className="px-2 py-1 bg-primary-100 dark:bg-primary-900/30 rounded-full">
                  <span className="text-xs font-semibold text-primary-600 dark:text-primary-400">Score ≥ 700</span>
                </div>
              </div>
            </div>
            <div className="p-4 bg-gradient-to-br from-primary-500 to-primary-600 rounded-2xl shadow-lg group-hover:shadow-xl transition-all duration-300">
              <ArrowTrendingUpIcon className="w-7 h-7 text-white" />
            </div>
          </div>
          <div className="h-1 bg-gradient-to-r from-primary-500 to-primary-600 rounded-full"></div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="group premium-card rounded-2xl p-8 hover-lift interactive-element"
        >
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-2 h-2 bg-accent-500 rounded-full"></div>
                <p className="text-sm font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wide">
                  Total Revenue
                </p>
              </div>
              <p className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                ${(totalRevenue / 1000000).toFixed(1)}M
              </p>
              <div className="flex items-center space-x-2">
                <div className="px-2 py-1 bg-accent-100 dark:bg-accent-900/30 rounded-full">
                  <span className="text-xs font-semibold text-accent-600 dark:text-accent-400">Monthly</span>
                </div>
              </div>
            </div>
            <div className="p-4 bg-gradient-to-br from-accent-500 to-accent-600 rounded-2xl shadow-lg group-hover:shadow-xl transition-all duration-300">
              <BanknotesIcon className="w-7 h-7 text-white" />
            </div>
          </div>
          <div className="h-1 bg-gradient-to-r from-accent-500 to-accent-600 rounded-full"></div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="group premium-card rounded-2xl p-8 hover-lift interactive-element"
        >
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-2 h-2 bg-amber-500 rounded-full"></div>
                <p className="text-sm font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wide">
                  Average Rating
                </p>
              </div>
              <p className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                {avgRating.toFixed(1)}
              </p>
              <div className="flex items-center space-x-2">
                <div className="flex items-center space-x-1">
                  {[...Array(5)].map((_, i) => (
                    <StarIcon 
                      key={i} 
                      className={`w-3 h-3 ${i < Math.floor(avgRating) ? 'text-amber-400 fill-current' : 'text-slate-300'}`} 
                    />
                  ))}
                </div>
                <span className="text-xs text-slate-500 dark:text-slate-500">out of 5.0</span>
              </div>
            </div>
            <div className="p-4 bg-gradient-to-br from-amber-500 to-amber-600 rounded-2xl shadow-lg group-hover:shadow-xl transition-all duration-300">
              <StarIcon className="w-7 h-7 text-white" />
            </div>
          </div>
          <div className="h-1 bg-gradient-to-r from-amber-500 to-amber-600 rounded-full"></div>
        </motion.div>
      </div>

      {/* Premium Credit Score Distribution */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="premium-card rounded-3xl p-10 hover-lift"
      >
        <div className="flex items-center justify-between mb-10">
          <div className="space-y-2">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-xl">
                <ChartBarIcon className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Credit Score Distribution</h3>
            </div>
            <p className="text-slate-600 dark:text-slate-400 text-lg">
              Real-time analysis across {selectedView === 'overview' ? 'all partners' : selectedView}
            </p>
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-slate-500">Drivers</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-emerald-500 rounded-full"></div>
                <span className="text-slate-500">Merchants</span>
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-slate-900 dark:text-slate-100">
              {filteredUsers.length.toLocaleString()}
            </div>
            <div className="text-sm text-slate-500">Total Records</div>
          </div>
        </div>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={scoreRanges} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis 
                dataKey="range" 
                tick={{ fill: '#64748b', fontSize: 12 }}
                axisLine={{ stroke: '#cbd5e1' }}
              />
              <YAxis 
                tick={{ fill: '#64748b', fontSize: 12 }}
                axisLine={{ stroke: '#cbd5e1' }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              {selectedView === 'overview' ? (
                <>
                  <Bar dataKey="drivers" fill="#0ea5e9" name="Drivers" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="merchants" fill="#22c55e" name="Merchants" radius={[4, 4, 0, 0]} />
                </>
              ) : (
                <Bar dataKey="count" fill={selectedView === 'drivers' ? '#0ea5e9' : '#22c55e'} name={selectedView === 'drivers' ? 'Drivers' : 'Merchants'} radius={[4, 4, 0, 0]} />
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Age Group Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="premium-card rounded-2xl p-8"
        >
          <div className="mb-6">
            <h3 className="text-xl font-bold text-slate-900">Age Group Distribution</h3>
            <p className="text-slate-600 mt-1">Partner distribution by age groups</p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={ageDistribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  dataKey="age" 
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  axisLine={{ stroke: '#cbd5e1' }}
                />
                <YAxis 
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  axisLine={{ stroke: '#cbd5e1' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                />
                {selectedView === 'overview' ? (
                  <>
                    <Area 
                      type="monotone" 
                      dataKey="drivers" 
                      stackId="1" 
                      stroke="#0ea5e9" 
                      fill="#0ea5e9"
                      fillOpacity={0.6}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="merchants" 
                      stackId="1" 
                      stroke="#22c55e" 
                      fill="#22c55e"
                      fillOpacity={0.6}
                    />
                  </>
                ) : (
                  <Area 
                    type="monotone" 
                    dataKey="count" 
                    stroke={selectedView === 'drivers' ? '#0ea5e9' : '#22c55e'} 
                    fill={selectedView === 'drivers' ? '#0ea5e9' : '#22c55e'}
                    fillOpacity={0.6}
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Performance Correlation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="premium-card rounded-2xl p-8"
        >
          <div className="mb-6">
            <h3 className="text-xl font-bold text-slate-900">Rating vs Credit Score</h3>
            <p className="text-slate-600 mt-1">Correlation between ratings and credit scores</p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  dataKey="rating" 
                  name="Rating" 
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  axisLine={{ stroke: '#cbd5e1' }}
                />
                <YAxis 
                  dataKey="creditScore" 
                  name="Credit Score" 
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  axisLine={{ stroke: '#cbd5e1' }}
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #e2e8f0',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                />
                <Scatter 
                  dataKey="creditScore" 
                  fill={selectedView === 'drivers' ? '#0ea5e9' : selectedView === 'merchants' ? '#22c55e' : '#10b981'}
                  fillOpacity={0.6}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Advanced Analytics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Risk Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.7 }}
          className="premium-card rounded-2xl p-8"
        >
          <div className="mb-6">
            <h3 className="text-xl font-bold text-slate-900">Risk Distribution</h3>
            <p className="text-slate-600 mt-1">Credit risk categorization</p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ risk, percent }) => `${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Earnings vs Credit Score */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8 }}
          className="premium-card rounded-2xl p-8"
        >
          <div className="mb-6">
            <h3 className="text-xl font-bold text-slate-900">Earnings Correlation</h3>
            <p className="text-slate-600 mt-1">Credit score vs earnings relationship</p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={earningsCorrelation}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  dataKey="creditScore" 
                  name="Credit Score" 
                  tick={{ fill: '#64748b', fontSize: 12 }}
                />
                <YAxis 
                  dataKey="earnings" 
                  name="Earnings" 
                  tick={{ fill: '#64748b', fontSize: 12 }}
                />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter 
                  dataKey="earnings" 
                  fill="#10b981"
                  fillOpacity={0.6}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Digital Payment Adoption */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.9 }}
          className="premium-card rounded-2xl p-8"
        >
          <div className="mb-6">
            <h3 className="text-xl font-bold text-slate-900">Digital Payment Adoption</h3>
            <p className="text-slate-600 mt-1">Adoption rate by age group</p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={digitalPaymentData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  dataKey="age" 
                  tick={{ fill: '#64748b', fontSize: 12 }}
                />
                <YAxis 
                  tick={{ fill: '#64748b', fontSize: 12 }}
                  domain={[0, 1]}
                />
                <Tooltip 
                  formatter={(value) => [`${(Number(value) * 100).toFixed(1)}%`, 'Adoption Rate']}
                />
                <Line 
                  type="monotone" 
                  dataKey="adoption" 
                  stroke="#22c55e" 
                  strokeWidth={3}
                  dot={{ fill: '#22c55e', strokeWidth: 2, r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Top Performers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {(selectedView === 'overview' || selectedView === 'drivers') && topDrivers.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 1.0 }}
            className="premium-card rounded-2xl p-8"
          >
            <div className="flex items-center space-x-3 mb-6">
              <TruckIcon className="w-6 h-6 text-accent-500" />
              <h3 className="text-xl font-bold text-slate-900">Top Drivers</h3>
            </div>
            <div className="space-y-4">
              {topDrivers.map((driver, index) => (
                <div key={driver.partner_id} className="flex items-center justify-between p-4 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors">
                  <div className="flex items-center space-x-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-accent-500 to-accent-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      {index + 1}
                    </div>
                    <div>
                      <div className="font-semibold text-slate-900">Driver #{driver.partner_id}</div>
                      <div className="text-sm text-slate-600">
                        Rating: {driver.average_rating.toFixed(1)} • {driver.tenure_months} months
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-bold text-accent-600">{driver.credit_score}</div>
                    <div className="text-sm text-slate-500">Credit Score</div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {(selectedView === 'overview' || selectedView === 'merchants') && topMerchants.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 1.1 }}
            className="premium-card rounded-2xl p-8"
          >
            <div className="flex items-center space-x-3 mb-6">
              <BuildingStorefrontIcon className="w-6 h-6 text-primary-500" />
              <h3 className="text-xl font-bold text-slate-900">Top Merchants</h3>
            </div>
            <div className="space-y-4">
              {topMerchants.map((merchant, index) => (
                <div key={merchant.partner_id} className="flex items-center justify-between p-4 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors">
                  <div className="flex items-center space-x-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      {index + 1}
                    </div>
                    <div>
                      <div className="font-semibold text-slate-900">Merchant #{merchant.partner_id}</div>
                      <div className="text-sm text-slate-600">
                        Rating: {merchant.average_rating.toFixed(1)} • {merchant.tenure_months} months
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-bold text-primary-600">{merchant.credit_score}</div>
                    <div className="text-sm text-slate-500">Credit Score</div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}