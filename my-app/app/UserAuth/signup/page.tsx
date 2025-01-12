'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import Background from '@/components/Background'
import Sidebar from '@/components/Sidebar'
import Link from 'next/link'  // Add this import

export default function SignupPage() {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [agreeTerms, setAgreeTerms] = useState(false)

  const router = useRouter()  // Hook for navigation

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Redirect to the login page after form submission
    router.push('/UserAuth/login')
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
          className="text-4xl font-bold text-center mb-6 text-pink-600"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Join the Fun!
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
              className="w-full px-4 py-3 text-lg border-2 border-pink-300 rounded-full focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent"
            />
          </motion.div>
          <motion.div 
            className="space-y-2"
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
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
              className="w-full px-4 py-3 text-lg border-2 border-pink-300 rounded-full focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent"
            />
          </motion.div>
          <motion.div 
            className="space-y-2"
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.5 }}
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
              className="w-full px-4 py-3 text-lg border-2 border-pink-300 rounded-full focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent"
            />
          </motion.div>
          <motion.div 
            className="space-y-2"
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            <label htmlFor="confirmPassword" className="block text-lg font-medium text-gray-700">
              Confirm Password
            </label>
            <Input
              id="confirmPassword"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              className="w-full px-4 py-3 text-lg border-2 border-pink-300 rounded-full focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent"
            />
          </motion.div>
          <motion.div 
            className="flex items-center space-x-2"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.7 }}
          >
            <Checkbox 
              id="terms" 
              checked={agreeTerms}
              onCheckedChange={(checked) => setAgreeTerms(checked as boolean)}
            />
            <label
              htmlFor="terms"
              className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
            >
              I promise to be nice and have fun!
            </label>
          </motion.div>
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.8 }}
          >
            <Button type="submit" className="w-full bg-green-400 hover:bg-green-500 text-blue-800 font-bold text-xl py-3 px-4 rounded-full transition duration-300 transform hover:scale-105">
              Let's Play!
            </Button>
          </motion.div>
        </form>
        <motion.p 
          className="mt-6 text-center text-lg"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          Already a friend?{' '}
          <Link href="/UserAuth/login" className="text-blue-600 hover:underline font-bold">
            LogIn
          </Link>
        </motion.p>
      </motion.div>
    </div>
  )
}
