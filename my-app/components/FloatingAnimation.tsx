import React from 'react'
import { motion } from 'framer-motion'

interface FloatingAnimationProps {
  children: React.ReactNode
  animation: 'float' | 'twinkle'
}

export default function FloatingAnimation({ children, animation }: FloatingAnimationProps) {
  const animationVariants = {
    float: {
      y: ['0%', '-20%', '0%'],
      transition: {
        duration: 3,
        repeat: Infinity,
        repeatType: 'reverse',
      },
    },
    twinkle: {
      opacity: [1, 0.5, 1],
      scale: [1, 1.2, 1],
      transition: {
        duration: 2,
        repeat: Infinity,
        repeatType: 'reverse',
      },
    },
  }

  return (
    <motion.div
      className="absolute"
      style={{
        top: `${Math.random() * 80}%`,
        left: `${Math.random() * 80}%`,
      }}
      animate={animationVariants[animation]}
    >
      {children}
    </motion.div>
  )
}

