'use client'

import { useTheme } from '@/contexts/ThemeContext'
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline'

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()

  return (
    <button
      onClick={toggleTheme}
      className="p-2 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors shadow-sm"
      aria-label="Toggle theme"
    >
      {theme === 'light' ? (
        <MoonIcon className="w-5 h-5 text-slate-600 dark:text-slate-400" />
      ) : (
        <SunIcon className="w-5 h-5 text-slate-600 dark:text-slate-400" />
      )}
    </button>
  )
}