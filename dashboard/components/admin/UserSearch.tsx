'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  MagnifyingGlassIcon,
  UserIcon,
  TruckIcon,
  BuildingStorefrontIcon,
  EyeIcon
} from '@heroicons/react/24/outline'
import { searchUsers } from '@/lib/dataService'

export default function UserSearch() {
  const [searchTerm, setSearchTerm] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedType, setSelectedType] = useState<'Driver' | 'Merchant'>('Driver')
  const [selectedUser, setSelectedUser] = useState<any>(null)

  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      // Load all users if no search term
      const results = await searchUsers('', selectedType)
      setSearchResults(results)
      return
    }
    
    setLoading(true)
    try {
      const results = await searchUsers(searchTerm, selectedType)
      setSearchResults(results)
    } catch (error) {
      console.error('Search error:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleUserSelect = (user: any) => {
    setSelectedUser(user)
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
      {/* Search Interface */}
      <div className="bg-white rounded-xl p-6 card-shadow-lg">
        <div className="flex items-center space-x-3 mb-6">
          <MagnifyingGlassIcon className="w-6 h-6 text-primary-500" />
          <h3 className="text-xl font-semibold text-gray-900">User Search & Management</h3>
        </div>

        <div className="space-y-4">
          <div className="flex space-x-4">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Search Users
              </label>
              <div className="relative">
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Enter Partner ID or search term..."
                  className="w-full pl-4 pr-12 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 bg-white text-slate-900 placeholder-slate-500 transition-all"
                  style={{ color: '#1e293b !important', backgroundColor: '#ffffff !important' }}
                />
                <button
                  onClick={handleSearch}
                  disabled={loading}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 p-2 text-slate-400 hover:text-primary-500 transition-colors"
                >
                  <MagnifyingGlassIcon className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                User Type
              </label>
              <div className="flex space-x-2">
                <button
                  onClick={() => setSelectedType('Driver')}
                  className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-medium transition-all ${
                    selectedType === 'Driver'
                      ? 'bg-accent-500 text-white shadow-sm'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  <TruckIcon className="w-4 h-4" />
                  <span>Drivers</span>
                </button>
                <button
                  onClick={() => setSelectedType('Merchant')}
                  className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-medium transition-all ${
                    selectedType === 'Merchant'
                      ? 'bg-primary-500 text-white shadow-sm'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  <BuildingStorefrontIcon className="w-4 h-4" />
                  <span>Merchants</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Search Results */}
        <div className="bg-white rounded-xl p-6 card-shadow-lg">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">
            Search Results ({searchResults.length})
          </h4>

          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500 mx-auto"></div>
              <p className="text-gray-600 mt-2">Searching...</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {searchResults.map((user) => (
                <div
                  key={user.partner_id}
                  className="p-4 border border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors cursor-pointer"
                  onClick={() => handleUserSelect(user)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex-shrink-0">
                        {user.partner_type === 'Driver' ? (
                          <TruckIcon className="w-5 h-5 text-gray-600" />
                        ) : (
                          <BuildingStorefrontIcon className="w-5 h-5 text-gray-600" />
                        )}
                      </div>
                      <div>
                        <div className="font-medium text-gray-900">
                          {user.partner_type} #{user.partner_id}
                        </div>
                        <div className="text-sm text-gray-600">
                          Rating: {user.average_rating?.toFixed(1)} | 
                          Tenure: {user.tenure_months} months
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className={`px-2 py-1 rounded text-sm font-semibold ${getScoreColor(user.credit_score)}`}>
                        {user.credit_score}
                      </div>
                      <EyeIcon className="w-4 h-4 text-gray-400" />
                    </div>
                  </div>
                </div>
              ))}

              {searchResults.length === 0 && !loading && (
                <div className="text-center py-8 text-gray-500">
                  No users found. Try a different search term or load all users.
                </div>
              )}
            </div>
          )}
        </div>

        {/* User Details */}
        <div className="bg-white rounded-xl p-6 card-shadow-lg">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">User Details</h4>

          {selectedUser ? (
            <div className="space-y-4">
              {/* User Header */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    {selectedUser.partner_type === 'Driver' ? (
                      <TruckIcon className="w-6 h-6 text-primary-600" />
                    ) : (
                      <BuildingStorefrontIcon className="w-6 h-6 text-primary-600" />
                    )}
                  </div>
                  <div>
                    <div className="font-semibold text-gray-900">
                      {selectedUser.partner_type} #{selectedUser.partner_id}
                    </div>
                    <div className="text-sm text-gray-600">
                      {selectedUser.location} • {selectedUser.age_group} • {selectedUser.gender}
                    </div>
                  </div>
                </div>
                <div className={`px-3 py-2 rounded-lg text-lg font-bold ${getScoreColor(selectedUser.credit_score)}`}>
                  {selectedUser.credit_score}
                </div>
              </div>

              {/* Key Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <div className="text-lg font-semibold text-blue-900">
                    {selectedUser.average_rating?.toFixed(1)}
                  </div>
                  <div className="text-sm text-blue-700">Rating</div>
                </div>
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <div className="text-lg font-semibold text-green-900">
                    {selectedUser.tenure_months}
                  </div>
                  <div className="text-sm text-green-700">Months</div>
                </div>
              </div>

              {/* All Parameters */}
              <div className="max-h-64 overflow-y-auto space-y-2">
                {Object.entries(selectedUser)
                  .filter(([key]) => !['partner_id', 'partner_type'].includes(key))
                  .map(([key, value]) => (
                    <div key={key} className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-sm font-medium text-gray-600 capitalize">
                        {key.replace(/_/g, ' ')}
                      </span>
                      <span className="text-sm text-gray-900">
                        {formatValue(key, value)}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <UserIcon className="w-12 h-12 mx-auto mb-4 text-gray-300" />
              <p>Select a user from the search results to view details</p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}