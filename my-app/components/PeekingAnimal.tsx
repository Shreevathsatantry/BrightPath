import React from 'react'
import { motion } from 'framer-motion'

type AnimalType = 'monkey' | 'elephant' | 'giraffe' | 'lion'

const animalEmojis: Record<AnimalType, string> = {
  monkey: '🐵',
  elephant: '🐘',
  giraffe: '🦒',
  lion: '🦁',
}

interface PeekingAnimalProps {
  type: AnimalType
  position: 'top' | 'right' | 'bottom' | 'left'
}

export default function PeekingAnimal({ type, position }: PeekingAnimalProps) {
  const positionStyles = {
    top: { top: -10, left: '50%', transform: 'translateX(-50%)' },
    right: { top: '50%', right: -10, transform: 'translateY(-50%)' },
    bottom: { bottom: -10, left: '50%', transform: 'translateX(-50%)' },
    left: { top: '50%', left: -10, transform: 'translateY(-50%)' },
  }

  return (
    <motion.div
      className="absolute z-10 text-4xl"
      style={positionStyles[position]}
      initial={{ opacity: 0, ...getInitialPosition(position) }}
      animate={{ opacity: 1, ...getFinalPosition(position) }}
      transition={{ 
        type: 'spring', 
        stiffness: 100, 
        damping: 10, 
        repeat: Infinity, 
        repeatType: 'reverse',
        duration: 2,
        repeatDelay: Math.random() * 2 + 1
      }}
    >
      {animalEmojis[type]}
    </motion.div>
  )
}

function getInitialPosition(position: 'top' | 'right' | 'bottom' | 'left') {
  switch (position) {
    case 'top': return { y: -20 }
    case 'right': return { x: 20 }
    case 'bottom': return { y: 20 }
    case 'left': return { x: -20 }
  }
}

function getFinalPosition(position: 'top' | 'right' | 'bottom' | 'left') {
  switch (position) {
    case 'top': return { y: 10 }
    case 'right': return { x: -10 }
    case 'bottom': return { y: -10 }
    case 'left': return { x: 10 }
  }
}

