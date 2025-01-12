import React from 'react'
import { motion } from 'framer-motion'

interface AnimatedCharacterProps {
  character: string
  style: React.CSSProperties
}

export default function AnimatedCharacter({ character, style }: AnimatedCharacterProps) {
  return (
    <motion.div
      className="absolute text-6xl"
      style={style}
      animate={{
        y: ['0%', '-20%', '0%'],
      }}
      transition={{
        duration: 2,
        repeat: Infinity,
        repeatType: 'reverse',
      }}
    >
      {character}
    </motion.div>
  )
}

