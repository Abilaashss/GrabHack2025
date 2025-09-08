'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { MagnifyingGlassIcon, UserCircleIcon } from '@heroicons/react/24/outline'
import { searchUsers } from '@/lib/dataService'

interface UserLoginProps {
  onLogin: (user: any) => void
}

export default function UserLogin({ onLogin }: UserLoginProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedType, setSelectedType] = useState<'Driver' | 'Merchant'>('Driver')

  const handleSearch = async () => {
    if (!searchTerm.trim()) return
    
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
    onLogin(user)
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="bg-white rounded-2xl p-8 card-shadow-lg max-w-md w-full"
      >
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-grab-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <UserCircleIcon className="w-8 h-8 text-grab-600" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Access Your Credit Score
          </h2>
          <p className="text-gray-600">
            Enter your Partner ID to view your credit score and details
          </p>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Account Type
            </label>
            <div className="flex space-x-2">
              <button
                onClick={() => setSelectedType('Driver')}
                className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
                  selectedType === 'Driver'
                    ? 'bg-grab-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Driver
              </button>
              <button
                onClick={() => setSelectedType('Merchant')}
                className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
                  selectedType === 'Merchant'
                    ? 'bg-grab-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Merchant
              </button>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Partner ID
            </label>
            <div className="relative">
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Enter your Partner ID"
                className="w-full pl-4 pr-12 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-grab-500 focus:border-transparent"
              />
              <button
                onClick={handleSearch}
                disabled={loading}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-gray-400 hover:text-grab-500"
              >
                <MagnifyingGlassIcon className="w-5 h-5" />
              </button>
            </div>
          </div>

          {loading && (
            <div className="text-center py-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-grab-500 mx-auto"></div>
            </div>
          )}

          {searchResults.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-700">Select your account:</h3>
              {searchResults.map((user) => (
                <button
                  key={user.partner_id}
                  onClick={() => handleUserSelect(user)}
                  className="w-full p-3 text-left border border-gray-200 rounded-lg hover:border-grab-500 hover:bg-grab-50 transition-colors"
                >
                  <div className="font-medium text-gray-900">
                    ID: {user.partner_id} - {user.partner_type}
                  </div>
                  <div className="text-sm text-gray-600">
                    Rating: {user.average_rating?.toFixed(1)} | 
                    Credit Score: {user.credit_score}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </motion.div>
    </div>
  )
}