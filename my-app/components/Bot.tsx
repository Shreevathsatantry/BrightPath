import React from 'react'
import { motion } from 'framer-motion'

interface BotProps {
  isCorrect: boolean | null
}

export default function Bot({ isCorrect }: BotProps) {
  const getBotEmotion = () => {
    if (isCorrect === null) return '😊'
    return isCorrect ? '😄' : '😢'
  }

  const getBotColor = () => {
    if (isCorrect === null) return 'bg-blue-500'
    return isCorrect ? 'bg-green-500' : 'bg-red-500'
  }

  return (
    <motion.div
      className={`w-24 h-24 ${getBotColor()} rounded-full overflow-hidden border-4 border-white shadow-lg flex items-center justify-center`}
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      transition={{ type: "spring", stiffness: 260, damping: 20 }}
    >
      <span className="text-4xl">{getBotEmotion()}</span>
    </motion.div>
  )
}

