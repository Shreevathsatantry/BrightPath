import React from 'react'
import { motion } from 'framer-motion'

interface TVBotProps {
  question: string
}

export default function TVBot({ question }: TVBotProps) {
  return (
    <motion.div
      className="bg-gray-800 p-4 rounded-xl shadow-lg w-64 h-48 flex flex-col items-center justify-center"
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <div className="bg-blue-400 w-48 h-32 rounded-lg overflow-hidden flex items-center justify-center relative">
        <motion.div
          className="absolute w-full h-full bg-blue-500"
          animate={{
            opacity: [0.5, 1, 0.5],
            scale: [1, 1.1, 1],
          }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        <motion.div
          className="z-10 text-white text-4xl font-bold"
          animate={{ rotate: [0, 5, -5, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          🤖
        </motion.div>
      </div>
      <motion.p
        className="mt-2 text-white text-center font-semibold"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        {question}
      </motion.p>
    </motion.div>
  )
}

