'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import UserDashboard from '@/components/user/UserDashboard'
import UserLogin from '@/components/user/UserLogin'

export default function UserPage() {
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Check if user is already logged in (from localStorage)
    const savedUser = localStorage.getItem('currentUser')
    if (savedUser) {
      setUser(JSON.parse(savedUser))
    }
    setLoading(false)
  }, [])

  const handleLogin = (userData: any) => {
    setUser(userData)
    localStorage.setItem('currentUser', JSON.stringify(userData))
  }

  const handleLogout = () => {
    setUser(null)
    localStorage.removeItem('currentUser')
  }

  if (loading) {
    return (
      <div className="min-h-screen gradient-bg flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-500"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen gradient-bg">
      {user ? (
        <UserDashboard user={user} onLogout={handleLogout} />
      ) : (
        <UserLogin onLogin={handleLogin} />
      )}
    </div>
  )
}