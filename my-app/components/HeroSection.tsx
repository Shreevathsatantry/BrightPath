import React from 'react'
import { motion } from 'framer-motion'
import Logo from './Logo'

export default function HeroSection() {
  return (
    <motion.div
      className="text-center"
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Logo />
      <h1 className="text-5xl font-bold mb-4 text-blue-800">Welcome to BrightPath!</h1>
      <p className="text-2xl mb-8 text-blue-600">
        Guiding children with autism to learn, grow, and shine!
      </p>
      <motion.button
        className="bg-yellow-400 text-blue-800 px-8 py-4 rounded-full text-2xl font-bold"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
      >
        Start Your Journey
      </motion.button>
    </motion.div>
  )
}

