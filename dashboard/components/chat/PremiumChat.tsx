'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ChatBubbleLeftRightIcon,
  PaperAirplaneIcon,
  SparklesIcon,
  XMarkIcon,
  LightBulbIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon
} from '@heroicons/react/24/outline'
import { ChatMessage, LLMResponse } from '@/lib/llmService'

interface PremiumChatProps {
  userRole: 'user' | 'admin'
  userType?: 'Driver' | 'Merchant'
  userId?: string
  isOpen: boolean
  onToggle: () => void
}

export default function PremiumChat({ userRole, userType, userId, isOpen, onToggle }: PremiumChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (isOpen) {
      loadSuggestions()
      inputRef.current?.focus()
    }
  }, [isOpen, userRole, userType])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const loadSuggestions = async () => {
    try {
      const params = new URLSearchParams({
        userRole,
        ...(userType && { userType }),
        ...(userId && { userId })
      })
      
      const response = await fetch(`/api/chat?${params}`)
      const data = await response.json()
      
      if (data.success) {
        setSuggestions(data.data.suggestions)
      }
    } catch (error) {
      console.error('Failed to load suggestions:', error)
    }
  }

  const sendMessage = async (message: string) => {
    if (!message.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      // Get browser context for enhanced intelligence
      const browserContext = {
        url: window.location.href,
        pathname: window.location.pathname,
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight
        }
      }

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          userRole,
          userType,
          userId,
          browserContext
        })
      })

      const data = await response.json()

      if (data.success) {
        const assistantMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.data.content,
          timestamp: new Date(),
          metadata: data.data.metadata
        }

        setMessages(prev => [...prev, assistantMessage])
      } else {
        throw new Error(data.error)
      }
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    sendMessage(suggestion)
  }

  const formatMessage = (content: string) => {
    // Split content into sections
    const sections = content.split(/(?=\*\*[^*]+\*\*)/g).filter(Boolean)
    
    return sections.map((section, index) => {
      if (section.startsWith('**') && section.includes('**')) {
        const [title, ...contentParts] = section.split('**')
        const sectionTitle = title.replace(/^\*\*/, '').replace(/\*\*$/, '')
        const sectionContent = contentParts.join('**')
        
        return (
          <div key={index} className="mb-4">
            <h4 className="font-semibold text-slate-900 dark:text-slate-100 mb-2 flex items-center space-x-2">
              {sectionTitle.toLowerCase().includes('insight') && <LightBulbIcon className="w-4 h-4 text-amber-500" />}
              {sectionTitle.toLowerCase().includes('recommendation') && <CheckCircleIcon className="w-4 h-4 text-emerald-500" />}
              {sectionTitle.toLowerCase().includes('risk') && <ExclamationTriangleIcon className="w-4 h-4 text-red-500" />}
              {sectionTitle.toLowerCase().includes('analysis') && <ChartBarIcon className="w-4 h-4 text-blue-500" />}
              <span>{sectionTitle}</span>
            </h4>
            <div className="text-slate-700 dark:text-slate-300 leading-relaxed">
              {sectionContent.split('\n').map((line, lineIndex) => {
                if (line.trim().startsWith('-') || line.trim().startsWith('•')) {
                  return (
                    <div key={lineIndex} className="flex items-start space-x-2 mb-1">
                      <div className="w-1.5 h-1.5 bg-primary-500 rounded-full mt-2 flex-shrink-0"></div>
                      <span>{line.trim().substring(1).trim()}</span>
                    </div>
                  )
                }
                return line.trim() ? <p key={lineIndex} className="mb-2">{line}</p> : null
              })}
            </div>
          </div>
        )
      }
      
      return (
        <p key={index} className="text-slate-700 dark:text-slate-300 leading-relaxed mb-2">
          {section}
        </p>
      )
    })
  }

  if (!isOpen) {
    return (
      <motion.button
        onClick={onToggle}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className="fixed bottom-6 right-6 z-50 w-16 h-16 bg-gradient-to-r from-primary-500 to-emerald-500 text-white rounded-full shadow-2xl hover:shadow-3xl transition-all duration-300 flex items-center justify-center group"
      >
        <ChatBubbleLeftRightIcon className="w-8 h-8 group-hover:scale-110 transition-transform" />
        <div className="absolute -top-2 -right-2 w-6 h-6 bg-gradient-to-r from-amber-400 to-orange-500 rounded-full flex items-center justify-center">
          <SparklesIcon className="w-4 h-4 text-white" />
        </div>
      </motion.button>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9, y: 20 }}
      className="fixed bottom-6 right-6 z-50 w-96 h-[600px] premium-card rounded-3xl overflow-hidden shadow-2xl"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-primary-500 to-emerald-500 p-6 text-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
              <SparklesIcon className="w-6 h-6" />
            </div>
            <div>
              <h3 className="font-bold text-lg">AI Assistant</h3>
              <p className="text-sm opacity-90">
                {userRole === 'admin' ? 'Admin Intelligence' : `${userType} Insights`}
              </p>
            </div>
          </div>
          <button
            onClick={onToggle}
            className="p-2 hover:bg-white/20 rounded-lg transition-colors bg-white/10 border border-white/20"
            title="Close Chat"
          >
            <XMarkIcon className="w-6 h-6 text-white" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 p-4 space-y-4 overflow-y-auto h-96">
        {messages.length === 0 && (
          <div className="text-center py-8">
            <SparklesIcon className="w-12 h-12 text-primary-500 mx-auto mb-4" />
            <h4 className="font-semibold text-slate-900 dark:text-slate-100 mb-2">
              Welcome to AI Assistant
            </h4>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
              Ask me anything about your credit data, performance metrics, or get personalized insights.
            </p>
            
            {/* Smart Suggestions */}
            <div className="space-y-2">
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                Smart Suggestions
              </p>
              {suggestions.slice(0, 3).map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="w-full text-left p-3 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg text-sm transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] p-4 rounded-2xl ${
                  message.role === 'user'
                    ? 'bg-gradient-to-r from-primary-500 to-emerald-500 text-white'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100'
                }`}
              >
                {message.role === 'user' ? (
                  <p className="leading-relaxed">{message.content}</p>
                ) : (
                  <div>
                    {formatMessage(message.content)}
                    
                    {message.metadata && (
                      <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700">
                        <div className="flex items-center space-x-2 text-xs text-slate-500 dark:text-slate-400">
                          <ClockIcon className="w-3 h-3" />
                          <span>Confidence: {((message.metadata.confidence || 0) * 100).toFixed(0)}%</span>
                          <span>•</span>
                          <span>Data: {(message.metadata.dataUsed || []).join(', ')}</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="bg-slate-100 dark:bg-slate-800 p-4 rounded-2xl">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-sm text-slate-600 dark:text-slate-400">AI is thinking...</span>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-slate-200 dark:border-slate-700">
        <div className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage(inputMessage)}
            placeholder="Ask me anything about your data..."
            className="flex-1 py-3 px-4 text-sm bg-white dark:bg-slate-800 border-2 border-slate-300 dark:border-slate-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-slate-900 dark:text-white placeholder-slate-500 dark:placeholder-slate-400"
            disabled={isLoading}
          />
          <button
            onClick={() => sendMessage(inputMessage)}
            disabled={!inputMessage.trim() || isLoading}
            className="p-3 bg-gradient-to-r from-primary-500 to-emerald-500 text-white rounded-xl hover:shadow-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </button>
        </div>
      </div>
    </motion.div>
  )
}