import React from 'react'
import { motion } from 'framer-motion'

export default function Logo() {
  return (
    <motion.div
      className="text-4xl font-bold text-blue-800 mb-4"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <span className="text-yellow-400">Bright</span>
      <span className="text-blue-600">Path</span>
    </motion.div>
  )
}

