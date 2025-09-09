'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import AdminDashboard from '@/components/admin/AdminDashboard'

export default function AdminPage() {
  const router = useRouter()

  const handleLogout = () => {
    // Clear any auth tokens/session data here
    router.push('/')
  }

  return (
    <div className="min-h-screen gradient-bg">
      <AdminDashboard onLogout={handleLogout} />
    </div>
  )
}