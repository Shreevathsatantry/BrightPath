'use client'

import { useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import Background from '@/components/Background'
// import AnimatedBackground from '@/components/AnimatedBackground'

export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    console.log('Login submitted:', { username, password })
  }

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* <AnimatedBackground /> */}
      <Background />
      <motion.div 
        className="bg-white p-8 rounded-3xl shadow-lg w-96 z-10"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <motion.h1 
          className="text-4xl font-bold text-center mb-6 text-blue-600"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Welcome, Friend HEEEEE!
        </motion.h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <motion.div 
            className="space-y-2"
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <label htmlFor="username" className="block text-lg font-medium text-gray-700">
              Username
            </label>
            <Input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="w-full px-4 py-3 text-lg border-2 border-blue-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </motion.div>
          <motion.div 
            className="space-y-2"
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <label htmlFor="password" className="block text-lg font-medium text-gray-700">
              Password
            </label>
            <Input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-4 py-3 text-lg border-2 border-blue-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </motion.div>
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <Button type="submit" className="w-full bg-yellow-400 hover:bg-yellow-500 text-blue-800 font-bold text-xl py-3 px-4 rounded-full transition duration-300 transform hover:scale-105">
              Let's Go!
            </Button>
          </motion.div>
        </form>
        <motion.p 
          className="mt-6 text-center text-lg"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          New here?{' '}
          <Link href="/signup" className="text-blue-600 hover:underline font-bold">
            Join the Fun!
          </Link>
        </motion.p>
      </motion.div>
    </div>
  )
}

