'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { MagnifyingGlassIcon, UserIcon, BuildingStorefrontIcon, ArrowRightIcon } from '@heroicons/react/24/outline'
import { useDebounce } from '@/lib/hooks'

interface User {
  partner_id: number
  partner_type: 'Driver' | 'Merchant'
  credit_score: number
  average_rating: number
  location: string
  tenure_months: number
  monthly_earnings?: number
  monthly_sales?: number
  completion_rate?: number
  order_acceptance_rate?: number
  total_trips?: number
  avg_order_value?: number
  age_group: string
  education_level: string
}

interface UserLoginProps {
  onLogin: (user: User) => void
}

export default function UserLogin({ onLogin }: UserLoginProps) {
  const [selectedType, setSelectedType] = useState<'Driver' | 'Merchant'>('Driver')
  const [searchId, setSearchId] = useState('')
  const [searchResults, setSearchResults] = useState<User[]>([])
  const [loading, setLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)

  // Debounced search function
  const debouncedSearch = useDebounce(async (searchTerm: string) => {
    if (searchTerm && searchTerm.length >= 1 && /^\d+$/.test(searchTerm.trim())) {
      await performSearch(searchTerm.trim())
    }
  }, 500)

  // Real-time search as user types
  const handleInputChange = async (value: string) => {
    setSearchId(value)
    setHasSearched(false)
    
    if (!value.trim()) {
      setSearchResults([])
      return
    }

    // Use debounced search for real-time results
    debouncedSearch(value)
  }

  const handleSearch = async () => {
    if (!searchId.trim()) {
      alert('Please enter a Partner ID')
      return
    }
    setHasSearched(true)
    await performSearch(searchId.trim())
  }

  const performSearch = async (searchTerm: string) => {
    if (loading) return
    
    setLoading(true)
    
    try {
      // Use the search API endpoint
      const response = await fetch(`/api/search?search=${encodeURIComponent(searchTerm)}&type=${selectedType}`)
      
      if (!response.ok) {
        throw new Error('Failed to fetch data')
      }
      
      const result = await response.json()
      console.log('Search API Response:', result)
      
      if (result.success && result.data) {
        // Prioritize exact matches
        const exactMatches = result.data.filter((user: User) => 
          user.partner_id.toString() === searchTerm
        )
        const partialMatches = result.data.filter((user: User) => 
          user.partner_id.toString().includes(searchTerm) && 
          user.partner_id.toString() !== searchTerm
        )
        
        const prioritizedResults = [...exactMatches, ...partialMatches]
        setSearchResults(prioritizedResults)
        
      } else {
        console.error('API Error:', result)
        setSearchResults([])
      }
    } catch (error) {
      console.error('Search failed:', error)
      // Fallback: try to get CSV data directly
      try {
        const endpoint = `/api/data/${selectedType.toLowerCase()}s`
        const response = await fetch(endpoint)
        const csvText = await response.text()
        
        // Parse CSV manually for search
        const lines = csvText.trim().split('\n')
        const headers = lines[0].split(',')
        const data = lines.slice(1).map(line => {
          const values = line.split(',')
          const obj: any = {}
          headers.forEach((header, index) => {
            obj[header.trim()] = values[index]?.trim() || ''
          })
          return obj
        })
        
        // Filter by search term with prioritized exact matches
        const exactMatches = data.filter((user: any) => 
          user.partner_id && user.partner_id === searchTerm
        )
        const partialMatches = data.filter((user: any) => 
          user.partner_id && user.partner_id.includes(searchTerm) && user.partner_id !== searchTerm
        )
        
        const filtered = [...exactMatches, ...partialMatches].slice(0, 10)
        
        // Convert to proper format
        const results = filtered.map((user: any) => ({
          partner_id: parseInt(user.partner_id) || 0,
          partner_type: user.partner_type || selectedType,
          credit_score: parseInt(user.credit_score) || 0,
          average_rating: parseFloat(user.average_rating) || 0,
          location: user.location || 'Unknown',
          tenure_months: parseInt(user.tenure_months) || 0,
          monthly_earnings: user.monthly_earnings ? parseInt(user.monthly_earnings) : undefined,
          monthly_sales: user.monthly_sales ? parseInt(user.monthly_sales) : undefined,
          completion_rate: user.completion_rate ? parseFloat(user.completion_rate) : undefined,
          order_acceptance_rate: user.order_acceptance_rate ? parseFloat(user.order_acceptance_rate) : undefined,
          total_trips: user.total_trips ? parseInt(user.total_trips) : undefined,
          avg_order_value: user.avg_order_value ? parseFloat(user.avg_order_value) : undefined,
          age_group: user.age_group || 'Unknown',
          education_level: user.education_level || 'Unknown'
        }))
        
        setSearchResults(results)
        
      } catch (fallbackError) {
        console.error('Fallback search failed:', fallbackError)
        setSearchResults([])
      }
    } finally {
      setLoading(false)
    }
  }

  const handleUserSelect = (user: User) => {
    onLogin(user)
  }

  return (
    <div className="min-h-screen gradient-bg flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <div className="premium-card p-8 text-center">
          <div className="mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-primary-500 to-emerald-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <UserIcon className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-2xl font-bold gradient-text mb-2">Partner Login</h1>
            <p className="text-slate-600 dark:text-slate-400">
              Access your credit intelligence dashboard
            </p>
          </div>

          {/* Partner Type Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Partner Type
            </label>
            <div className="flex bg-slate-200 dark:bg-slate-700 rounded-lg p-1">
              <button
                onClick={() => setSelectedType('Driver')}
                className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all duration-200 ${
                  selectedType === 'Driver'
                    ? 'bg-gradient-to-r from-primary-500 to-emerald-500 text-white shadow-lg'
                    : 'text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-300 dark:hover:bg-slate-600'
                }`}
              >
                🚗 Driver
              </button>
              <button
                onClick={() => setSelectedType('Merchant')}
                className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all duration-200 ${
                  selectedType === 'Merchant'
                    ? 'bg-gradient-to-r from-primary-500 to-emerald-500 text-white shadow-lg'
                    : 'text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-300 dark:hover:bg-slate-600'
                }`}
              >
                🏪 Merchant
              </button>
            </div>
          </div>

          {/* Search Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Partner ID
            </label>
            <div className="relative">
              <input
                type="text"
                value={searchId}
                onChange={(e) => handleInputChange(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder={`Enter your ${selectedType} ID (e.g., 109)`}
                className="w-full px-4 py-3 bg-white dark:bg-slate-800 border-2 border-slate-300 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-slate-900 dark:text-white placeholder-slate-500 dark:placeholder-slate-400"
                style={{ fontSize: '16px' }}
              />
              <button
                onClick={handleSearch}
                disabled={loading}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 p-2 text-slate-500 hover:text-primary-500 transition-colors bg-white dark:bg-slate-800 rounded-md"
              >
                <MagnifyingGlassIcon className="w-5 h-5" />
              </button>
            </div>
          </div>

          {loading && (
            <div className="text-center py-6 bg-slate-50 dark:bg-slate-800 rounded-lg border-2 border-slate-200 dark:border-slate-600">
              <div className="animate-spin rounded-full h-10 w-10 border-b-3 border-primary-500 mx-auto"></div>
              <p className="text-slate-700 dark:text-slate-300 mt-3 font-medium">
                Searching for {selectedType} #{searchId}...
              </p>
            </div>
          )}

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                Found {searchResults.length} {selectedType.toLowerCase()}(s):
              </h3>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {searchResults.map((user) => (
                  <button
                    key={user.partner_id}
                    onClick={() => handleUserSelect(user)}
                    className="w-full p-4 text-left bg-white dark:bg-slate-800 border-2 border-slate-200 dark:border-slate-600 rounded-lg hover:border-primary-500 hover:shadow-lg transition-all duration-300"
                  >
                    <div className="flex items-center space-x-3">
                      <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                        user.partner_type === 'Driver' 
                          ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' 
                          : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
                      }`}>
                        {user.partner_type === 'Driver' ? 
                          <UserIcon className="w-6 h-6" /> : 
                          <BuildingStorefrontIcon className="w-6 h-6" />
                        }
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-slate-900 dark:text-slate-100">
                          {user.partner_type} #{user.partner_id}
                        </div>
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                          📍 {user.location} • ⭐ Score: {user.credit_score}
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-500">
                          Rating: {user.average_rating.toFixed(1)}/5.0 • {user.tenure_months} months
                        </div>
                      </div>
                      <div className="text-primary-500">
                        <ArrowRightIcon className="w-5 h-5" />
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {searchResults.length === 0 && searchId && !loading && hasSearched && (
            <div className="text-center py-6 bg-red-50 dark:bg-red-900/20 rounded-lg border-2 border-red-200 dark:border-red-800">
              <div className="text-red-500 text-4xl mb-2">🔍</div>
              <p className="text-red-700 dark:text-red-300 font-medium">
                No {selectedType.toLowerCase()} found with ID "{searchId}"
              </p>
              <p className="text-red-600 dark:text-red-400 text-sm mt-1">
                Try a different ID or check the partner type
              </p>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  )
}