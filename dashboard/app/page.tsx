'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { UserCircleIcon, BuildingOfficeIcon } from '@heroicons/react/24/outline'
import Link from 'next/link'

export default function Home() {
  const [selectedRole, setSelectedRole] = useState<'user' | 'admin' | null>(null)

  return (
    <div className="min-h-screen gradient-bg flex items-center justify-center p-4">
      <div className="max-w-4xl w-full">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Grab Credit Score
            <span className="text-grab-500"> Dashboard</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Comprehensive credit scoring platform for drivers, merchants, and administrators
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8 max-w-3xl mx-auto">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-2xl p-8 card-shadow-lg border border-gray-100"
          >
            <div className="text-center">
              <div className="w-16 h-16 bg-grab-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <UserCircleIcon className="w-8 h-8 text-grab-600" />
              </div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Driver & Merchant Portal
              </h2>
              <p className="text-gray-600 mb-6">
                Check your credit score, view detailed parameters, and track your performance metrics
              </p>
              <Link
                href="/user"
                className="inline-flex items-center justify-center w-full px-6 py-3 bg-grab-500 text-white font-medium rounded-lg hover:bg-grab-600 transition-colors"
              >
                Access Portal
              </Link>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            whileHover={{ scale: 1.02 }}
            className="bg-white rounded-2xl p-8 card-shadow-lg border border-gray-100"
          >
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <BuildingOfficeIcon className="w-8 h-8 text-primary-600" />
              </div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Admin Dashboard
              </h2>
              <p className="text-gray-600 mb-6">
                Manage models, view analytics, search users, and monitor system performance
              </p>
              <Link
                href="/admin"
                className="inline-flex items-center justify-center w-full px-6 py-3 bg-primary-500 text-white font-medium rounded-lg hover:bg-primary-600 transition-colors"
              >
                Admin Access
              </Link>
            </div>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="text-center mt-12"
        >
          <div className="bg-white rounded-xl p-6 card-shadow max-w-2xl mx-auto">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Powered by Advanced ML Models
            </h3>
            <p className="text-gray-600">
              Our platform uses multiple machine learning algorithms including Random Forest, 
              Gradient Boosting, and Neural Networks to provide accurate credit scoring.
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  )
}