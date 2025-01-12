import React from 'react'
import { motion } from 'framer-motion'

const Cloud = ({ delay, scale = 1 }: { delay: number; scale?: number }) => (
  <motion.div
    className="absolute bg-white rounded-full opacity-80"
    style={{
      width: (Math.random() * 100 + 50) * scale,
      height: (Math.random() * 60 + 30) * scale,
    }}
    animate={{
      x: ['0%', '100%'],
      y: [0, Math.random() * 20 - 10],
      transition: {
        duration: Math.random() * 60 + 30,
        delay,
        repeat: Infinity,
        repeatType: 'loop',
        y: {
          duration: 5,
          repeat: Infinity,
          repeatType: 'reverse',
          ease: 'easeInOut'
        }
      },
    }}
  />
)

const Star = ({ x, y }: { x: number; y: number }) => (
  <motion.div
    className="absolute bg-yellow-200 rounded-full w-2 h-2"
    style={{ left: `${x}%`, top: `${y}%` }}
    animate={{
      scale: [1, 1.5, 1],
      opacity: [0.5, 1, 0.5],
    }}
    transition={{
      duration: 2,
      repeat: Infinity,
      repeatType: 'reverse',
      delay: Math.random() * 2,
    }}
  />
)

const Rainbow = () => (
  <motion.div
    className="absolute left-[5%] bottom-[20%] w-[90%] h-[300px] opacity-20"
    style={{
      background: 'linear-gradient(180deg, #ff0000, #ff7f00, #ffff00, #00ff00, #0000ff, #4b0082, #8b00ff)',
      borderRadius: '150px 150px 0 0',
    }}
    initial={{ opacity: 0 }}
    animate={{ opacity: 0.2 }}
    transition={{ duration: 2 }}
  />
)

const Trees = () => (
  <div className="absolute bottom-0 left-0 right-0 h-32">
    <motion.div
      className="absolute bottom-0 left-[10%] w-32 h-48 bg-green-700 rounded-t-full"
      animate={{ y: [0, -5, 0] }}
      transition={{ duration: 3, repeat: Infinity, repeatType: 'reverse' }}
    />
    <motion.div
      className="absolute bottom-0 left-[30%] w-24 h-40 bg-green-600 rounded-t-full"
      animate={{ y: [0, -3, 0] }}
      transition={{ duration: 2.5, repeat: Infinity, repeatType: 'reverse', delay: 0.5 }}
    />
    <motion.div
      className="absolute bottom-0 right-[20%] w-28 h-44 bg-green-800 rounded-t-full"
      animate={{ y: [0, -4, 0] }}
      transition={{ duration: 3.5, repeat: Infinity, repeatType: 'reverse', delay: 1 }}
    />
  </div>
)

export default function Background() {
  return (
    <div className="fixed inset-0 overflow-hidden bg-gradient-to-b from-blue-300 to-blue-100">
      <Rainbow />
      <div className="stars">
        {Array.from({ length: 20 }).map((_, i) => (
          <Star key={i} x={Math.random() * 100} y={Math.random() * 60} />
        ))}
      </div>
      <Cloud delay={0} scale={1.5} />
      <Cloud delay={5} />
      <Cloud delay={10} scale={0.8} />
      <Cloud delay={15} scale={1.2} />
      <Trees />
    </div>
  )
}

