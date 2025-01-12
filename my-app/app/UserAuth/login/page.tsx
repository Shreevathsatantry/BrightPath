'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation' // Import useRouter for redirection
import Link from 'next/link'
import { motion } from 'framer-motion'
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import Background from '@/components/Background'
import Sidebar from '@/components/Sidebar'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const router = useRouter() // Initialize the useRouter hook for redirection

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    console.log('Login submitted:', { email, password })
    
    // Simulating successful login and redirecting to homepage
    router.push('/') // This will navigate to the homepage after successful login
  }

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden">
      <Background />
      <Sidebar />
      <motion.div 
        className="bg-white bg-opacity-80 p-8 rounded-3xl shadow-lg w-96 z-10"
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
          Welcome, Friend!
        </motion.h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <motion.div 
            className="space-y-2"
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <label htmlFor="email" className="block text-lg font-medium text-gray-700">
              Email
            </label>
            <Input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
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
          <Link href="/UserAuth/signup" className="text-blue-600 hover:underline font-bold">
            SignUp
          </Link>
        </motion.p>
      </motion.div>
    </div>
  )
}
