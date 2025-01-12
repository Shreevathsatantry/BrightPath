'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Background from '../../../components/Background'
import Sidebar from '../../../components/Sidebar'
import ChatInterface from '../../../components/ChatInterface'
import { Star, Cloud, Sun } from 'lucide-react'

export default function ChatbotPage() {
  const [learningTopic, setLearningTopic] = useState<{ topic: string; subject: string; details: string } | null>(null)

  useEffect(() => {
    const storedTopic = localStorage.getItem('learningTopic')
    console.log("Chatbot: ", storedTopic)
    if (storedTopic) {
      setLearningTopic(JSON.parse(storedTopic))
    }
  }, [])

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
          className="max-w-4xl mx-auto bg-blue-100 rounded-3xl shadow-2xl p-8 border-4 border-yellow-300"
        >
          <motion.h1
            className="text-5xl font-bold text-center text-blue-600 mb-6 font-comic"
            initial={{ scale: 0 }}
            animate={{ scale: 1, rotate: [0, 2, -2, 0] }}
            transition={{ duration: 0.5, rotate: { repeat: Infinity, duration: 2 } }}
          >
            Your Learning Buddy
          </motion.h1>
          <p className="text-xl text-center text-blue-500 mb-8 font-comic">
            Here's your custom explanation for the topic you chose! Ask questions or use your voice to learn more.
          </p>
          {/* {learningTopic && (
            <div className="mb-8 p-4 bg-green-100 rounded-2xl border-2 border-green-300">
              <h2 className="text-2xl font-bold text-green-600 mb-2 font-comic">Your Topic: {learningTopic.topic}</h2>
              <p className="text-lg text-green-500 font-comic">Subject: {learningTopic.subject}</p>
              {learningTopic.details && (
                <p className="text-lg text-green-500 mt-2 font-comic">Additional Details: {learningTopic.details}</p>
              )}
            </div>
          )} */}
          <ChatInterface />
        </motion.div>
      </main>
    </div>
  )
}

