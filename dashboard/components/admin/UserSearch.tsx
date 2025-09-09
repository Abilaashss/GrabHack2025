'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  MagnifyingGlassIcon,
  UserIcon,
  TruckIcon,
  BuildingStorefrontIcon,
  EyeIcon
} from '@heroicons/react/24/outline'
import { searchUsers } from '@/lib/dataService'
import { useDebounce } from '@/lib/hooks'

export default function UserSearch() {
  const [searchTerm, setSearchTerm] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedType, setSelectedType] = useState<'Driver' | 'Merchant'>('Driver')
  const [selectedUser, setSelectedUser] = useState<any>(null)
  const [hasSearched, setHasSearched] = useState(false)

  // Debounced search function
  const debouncedSearch = useDebounce(async (term: string) => {
    if (term.length >= 1) {
      await performSearch(term)
    }
  }, 300)

  // Real-time search as user types
  const handleInputChange = async (value: string) => {
    setSearchTerm(value)
    setHasSearched(false)
    
    if (!value.trim()) {
      // Load some default users when search is cleared
      await loadDefaultUsers()
      return
    }

    // Use debounced search for performance
    debouncedSearch(value)
  }

  const loadDefaultUsers = async () => {
    try {
      const response = await fetch(`/api/search?search=&type=${selectedType}`)
      const result = await response.json()
      if (result.success) {
        setSearchResults(result.data.slice(0, 20)) // Show first 20 users
      }
    } catch (error) {
      console.error('Load default users error:', error)
    }
  }

  const handleSearch = async () => {
    setHasSearched(true)
    await performSearch(searchTerm)
  }

  const performSearch = async (term: string) => {
    if (loading) return
    
    setLoading(true)
    try {
      // Use the enhanced search API
      const response = await fetch(`/api/search?search=${encodeURIComponent(term)}&type=${selectedType}`)
      const result = await response.json()
      
      if (result.success) {
        setSearchResults(result.data)
      } else {
        console.error('Search API Error:', result)
        // Fallback to original search method
        const results = await searchUsers(term, selectedType)
        setSearchResults(results)
      }
    } catch (error) {
      console.error('Search error:', error)
      // Fallback to original search method
      try {
        const results = await searchUsers(term, selectedType)
        setSearchResults(results)
      } catch (fallbackError) {
        console.error('Fallback search error:', fallbackError)
        setSearchResults([])
      }
    } finally {
      setLoading(false)
    }
  }

  // Load default users on component mount and type change
  useEffect(() => {
    loadDefaultUsers()
  }, [selectedType])

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
      {/* Premium Search Interface */}
      <div className="premium-card rounded-3xl p-10 hover-lift">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-gradient-to-br from-primary-500 to-primary-600 rounded-2xl shadow-lg">
              <MagnifyingGlassIcon className="w-7 h-7 text-white" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Advanced User Search</h3>
              <p className="text-slate-600 dark:text-slate-400">Intelligent partner discovery and analysis</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
              {searchResults.length} Results
            </span>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex space-x-4">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Search Users
              </label>
              <div className="relative group">
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => handleInputChange(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Search by Partner ID, name, or any criteria..."
                  className="w-full pl-6 pr-16 py-4 text-lg bg-white dark:bg-slate-800 border-2 border-slate-300 dark:border-slate-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-slate-900 dark:text-white placeholder-slate-500 dark:placeholder-slate-400"
                />
                <button
                  onClick={handleSearch}
                  disabled={loading}
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 p-3 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-xl hover:shadow-lg transition-all duration-300 hover:scale-105 disabled:opacity-50"
                >
                  {loading ? (
                    <div className="loading-spinner w-5 h-5"></div>
                  ) : (
                    <MagnifyingGlassIcon className="w-5 h-5" />
                  )}
                </button>
                {/* Search suggestions overlay */}
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-primary-500/10 to-emerald-500/10 opacity-0 group-focus-within:opacity-100 transition-opacity duration-300 -z-10 blur-sm"></div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                User Type
              </label>
              <div className="flex space-x-2 p-2 bg-slate-100/50 dark:bg-slate-800/50 rounded-2xl backdrop-blur-sm">
                <button
                  onClick={() => setSelectedType('Driver')}
                  className={`flex items-center space-x-3 px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                    selectedType === 'Driver'
                      ? 'bg-gradient-to-r from-accent-500 to-accent-600 text-white shadow-lg hover:shadow-xl'
                      : 'text-slate-700 dark:text-slate-300 hover:bg-white dark:hover:bg-slate-700 hover:shadow-md'
                  }`}
                >
                  <TruckIcon className="w-5 h-5" />
                  <span>Drivers</span>
                  {selectedType === 'Driver' && (
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  )}
                </button>
                <button
                  onClick={() => setSelectedType('Merchant')}
                  className={`flex items-center space-x-3 px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                    selectedType === 'Merchant'
                      ? 'bg-gradient-to-r from-primary-500 to-primary-600 text-white shadow-lg hover:shadow-xl'
                      : 'text-slate-700 dark:text-slate-300 hover:bg-white dark:hover:bg-slate-700 hover:shadow-md'
                  }`}
                >
                  <BuildingStorefrontIcon className="w-5 h-5" />
                  <span>Merchants</span>
                  {selectedType === 'Merchant' && (
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  )}
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

              {searchResults.length === 0 && !loading && hasSearched && searchTerm && (
                <div className="text-center py-8 text-gray-500">
                  No users found for "{searchTerm}". Try a different search term.
                </div>
              )}
              
              {searchResults.length === 0 && !loading && !hasSearched && !searchTerm && (
                <div className="text-center py-8 text-gray-500">
                  Enter a Partner ID or search term to find users.
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