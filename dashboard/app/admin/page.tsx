'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import AdminDashboard from '@/components/admin/AdminDashboard'

export default function AdminPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <AdminDashboard />
    </div>
  )
}