import React from 'react'
import { motion } from 'framer-motion'

interface FeatureCardProps {
  title: string
  description: string
  icon: string
  color: string
}

export default function FeatureCard({ title, description, icon, color }: FeatureCardProps) {
  return (
    <motion.div
      className={`p-6 rounded-3xl shadow-lg ${color}`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <div className="text-4xl mb-4">{icon}</div>
      <h3 className="text-2xl font-bold mb-2">{title}</h3>
      <p>{description}</p>
    </motion.div>
  )
}

