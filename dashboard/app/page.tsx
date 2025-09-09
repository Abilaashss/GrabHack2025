'use client'

import { useState, useEffect } from 'react'
import { motion, useScroll, useTransform } from 'framer-motion'
import { 
  UserCircleIcon, 
  BuildingOfficeIcon, 
  ChartBarIcon,
  CpuChipIcon,
  ShieldCheckIcon,
  SparklesIcon,
  ArrowRightIcon,
  PlayIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'
import ThemeToggle from '@/components/ThemeToggle'

export default function Home() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const { scrollY } = useScroll()
  const y1 = useTransform(scrollY, [0, 300], [0, -50])
  const y2 = useTransform(scrollY, [0, 300], [0, -100])

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY })
    }
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  const features = [
    {
      icon: CpuChipIcon,
      title: "AI-Powered Analytics",
      description: "Advanced machine learning models with 99.99% accuracy",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: ShieldCheckIcon,
      title: "Fairness & Compliance",
      description: "Bias-free scoring across all demographic groups",
      color: "from-emerald-500 to-teal-500"
    },
    {
      icon: ChartBarIcon,
      title: "Real-Time Insights",
      description: "Live analytics with interactive visualizations",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: SparklesIcon,
      title: "LLM-Powered Chat",
      description: "Ask anything about your data with Llama 3-70B",
      color: "from-amber-500 to-orange-500"
    }
  ]

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-50 via-white to-emerald-50 dark:from-slate-900 dark:via-slate-800 dark:to-emerald-900"></div>
        
        {/* Floating Orbs */}
        <motion.div 
          className="absolute w-96 h-96 bg-gradient-to-r from-primary-500/20 to-emerald-500/20 rounded-full blur-3xl"
          style={{ 
            x: mousePosition.x * 0.02,
            y: mousePosition.y * 0.02,
            top: '10%',
            left: '10%'
          }}
        />
        <motion.div 
          className="absolute w-80 h-80 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-full blur-3xl"
          style={{ 
            x: mousePosition.x * -0.01,
            y: mousePosition.y * -0.01,
            bottom: '10%',
            right: '10%'
          }}
        />
        
        {/* Grid Pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(34,197,94,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(34,197,94,0.03)_1px,transparent_1px)] bg-[size:50px_50px]"></div>
      </div>

      {/* Header */}
      <header className="relative z-10 p-6">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center space-x-3"
          >
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-emerald-500 rounded-xl flex items-center justify-center">
              <SparklesIcon className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold gradient-text">Grab Credit AI</span>
          </motion.div>
          <ThemeToggle />
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative z-10 pt-20 pb-32">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-12"
          >
            <div className="inline-flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-primary-500/10 to-emerald-500/10 rounded-full border border-primary-500/20 mb-8">
              <SparklesIcon className="w-4 h-4 text-primary-600" />
              <span className="text-sm font-semibold text-primary-600 dark:text-primary-400">
                Powered by Hack Grabbers
              </span>
            </div>
            
            <h1 className="text-7xl font-bold mb-8 leading-tight">
              <span className="gradient-text">Next-Generation</span>
              <br />
              <span className="text-slate-900 dark:text-slate-100">Credit Intelligence</span>
            </h1>
            
            <p className="text-2xl text-slate-600 dark:text-slate-400 max-w-4xl mx-auto mb-12 leading-relaxed">
              Revolutionary AI-powered credit scoring platform with advanced machine learning, 
              real-time analytics, and intelligent chat assistance for Grab's ecosystem.
            </p>

            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
              <Link href="/user" className="group">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="premium-button flex items-center space-x-3 px-8 py-4 text-lg"
                >
                  <UserCircleIcon className="w-6 h-6" />
                  <span>Partner Portal</span>
                  <ArrowRightIcon className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </motion.button>
              </Link>
              
              <Link href="/admin" className="group">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="flex items-center space-x-3 px-8 py-4 text-lg bg-slate-900 dark:bg-white text-white dark:text-slate-900 rounded-xl font-semibold hover:bg-slate-800 dark:hover:bg-slate-100 transition-all duration-300"
                >
                  <BuildingOfficeIcon className="w-6 h-6" />
                  <span>Admin Dashboard</span>
                  <ArrowRightIcon className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </motion.button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="relative z-10 py-32">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-20"
          >
            <h2 className="text-5xl font-bold text-slate-900 dark:text-slate-100 mb-6">
              Cutting-Edge Features
            </h2>
            <p className="text-xl text-slate-600 dark:text-slate-400 max-w-3xl mx-auto">
              Experience the future of credit scoring with our advanced AI platform
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="group premium-card p-8 hover-lift interactive-element"
              >
                <div className={`w-16 h-16 bg-gradient-to-br ${feature.color} rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-4">
                  {feature.title}
                </h3>
                <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative z-10 py-32">
        <motion.div 
          style={{ y: y1 }}
          className="max-w-7xl mx-auto px-6"
        >
          <div className="premium-card p-16 text-center">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8 }}
            >
              <h2 className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-12">
                Trusted by Grab.
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
                {[
                  { value: "99.99%", label: "Model Accuracy" },
                  { value: "3,600+", label: "Active Partners" },
                  { value: "8", label: "ML Models" },
                  { value: "100%", label: "Bias-Free Scoring" }
                ].map((stat, index) => (
                  <motion.div
                    key={stat.label}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                    className="text-center"
                  >
                    <div className="text-4xl font-bold gradient-text mb-2">
                      {stat.value}
                    </div>
                    <div className="text-slate-600 dark:text-slate-400 font-medium">
                      {stat.label}
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>
        </motion.div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 py-32">
        <motion.div 
          style={{ y: y2 }}
          className="max-w-4xl mx-auto px-6 text-center"
        >
          <div className="premium-card p-16 bg-gradient-to-br from-primary-500/5 to-emerald-500/5 border border-primary-500/20">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <SparklesIcon className="w-16 h-16 text-primary-500 mx-auto mb-8" />
              <h2 className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-6">
                Ready to Experience the Future?
              </h2>
              <p className="text-xl text-slate-600 dark:text-slate-400 mb-12">
                Join thousands of partners already using our AI-powered credit intelligence platform
              </p>
              
              <div className="flex flex-col sm:flex-row gap-6 justify-center">
                <Link href="/user">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="premium-button flex items-center space-x-3 px-8 py-4 text-lg"
                  >
                    <PlayIcon className="w-6 h-6" />
                    <span>Get Started Now</span>
                  </motion.button>
                </Link>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 py-12 border-t border-slate-200/50 dark:border-slate-700/50">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-slate-600 dark:text-slate-400">
            © 2024 Grab Credit AI. Powered by advanced machine learning algorithms
          </p>
        </div>
      </footer>
    </div>
  )
}