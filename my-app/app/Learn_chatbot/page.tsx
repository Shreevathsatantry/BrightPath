'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import Background from '../../components/Background'
import Sidebar from '../../components/Sidebar'
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Star, Cloud, Sun } from 'lucide-react'

export default function InputPage() {
  const router = useRouter()
  const [topic, setTopic] = useState('')
  const [subject, setSubject] = useState('')
  const [details, setDetails] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    localStorage.setItem('learningTopic', JSON.stringify({ topic, subject, details }))
    router.push('/Learn_chatbot/chatbot')
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      <Background />
      <Sidebar />
      <main className="relative z-10 p-8 ml-10">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <motion.div
            className="absolute top-5 left-5 text-yellow-300"
            animate={{ rotate: 360, scale: [1, 1.2, 1] }}
            transition={{ duration: 5, repeat: Infinity }}
          >
            <Star size={24} />
          </motion.div>
          <motion.div
            className="absolute bottom-5 right-5 text-blue-300"
            animate={{ y: [0, -10, 0], rotate: [0, 10, -10, 0] }}
            transition={{ duration: 4, repeat: Infinity }}
          >
            <Cloud size={32} />
          </motion.div>
          <motion.div
            className="absolute top-1/2 right-5 text-pink-300"
            animate={{ rotate: 360, scale: [1, 1.1, 1] }}
            transition={{ duration: 10, repeat: Infinity }}
          >
            <Sun size={40} />
          </motion.div>
        </div>
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-2xl mx-auto bg-blue-100 rounded-3xl shadow-2xl p-8 border-4 border-yellow-300"
        >
          <motion.h1
            className="text-5xl font-bold text-center text-blue-600 mb-6 font-comic"
            initial={{ scale: 0 }}
            animate={{ scale: 1, rotate: [0, 2, -2, 0] }}
            transition={{ duration: 0.5, rotate: { repeat: Infinity, duration: 2 } }}
          >
            Learn Like an Explorer!
          </motion.h1>
          <p className="text-xl text-center text-blue-500 mb-8 font-comic">
            Dive into the world of learning with your favorite topics! Let's make learning fun and exciting.
          </p>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="topic" className="block text-lg font-medium text-blue-600 mb-1 font-comic">Topic</label>
              <Input
                id="topic"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Enter a topic (e.g., dinosaurs, space, rainbows)"
                required
                className="bg-white text-lg font-comic rounded-full p-2 border-2 border-blue-300 focus:border-pink-300 focus:ring focus:ring-pink-200 focus:ring-opacity-50"
              />
            </div>
            <div>
              <label htmlFor="subject" className="block text-lg font-medium text-blue-600 mb-1 font-comic">Subject</label>
              <Select onValueChange={setSubject} required>
                <SelectTrigger className="bg-white text-lg font-comic rounded-full border-2 border-blue-300 focus:border-green-300 focus:ring focus:ring-green-200 focus:ring-opacity-50">
                  <SelectValue placeholder="Choose a subject" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="science">Science</SelectItem>
                  <SelectItem value="history">History</SelectItem>
                  <SelectItem value="art">Art</SelectItem>
                  <SelectItem value="math">Math</SelectItem>
                  <SelectItem value="literature">Literature</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <label htmlFor="details" className="block text-lg font-medium text-blue-600 mb-1 font-comic">Additional Details</label>
              <Textarea
                id="details"
                value={details}
                onChange={(e) => setDetails(e.target.value)}
                placeholder="Add any extra details or specific questions you have"
                rows={4}
                className="bg-white text-lg font-comic rounded-2xl p-2 border-2 border-blue-300 focus:border-yellow-300 focus:ring focus:ring-yellow-200 focus:ring-opacity-50"
              />
            </div>
            <Button
              type="submit"
              className="w-full bg-yellow-400 hover:bg-yellow-500 text-blue-700 font-bold py-3 px-4 rounded-full text-2xl transition-all duration-300 transform hover:scale-105 hover:rotate-1 font-comic"
            >
              Let's Go on an Adventure!
            </Button>
          </form>
        </motion.div>
      </main>
    </div>
  )
}

