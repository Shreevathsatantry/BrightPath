'use client'

import React, { useRef, useState } from 'react'
// import React, { useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import Background from '../../components/Background'  // Correct path to Background
import Sidebar from '../../components/Sidebar'  // Correct path to Sidebar
import DrawableCanvas from '../../components/DrawableCanvas'  // Correct path to DrawableCanvas
import StoryInput from '../../components/StoryInput'  // Correct path to StoryInput
import Book from '../../components/Book'  // Correct path to Book
import { Button } from '@/components/ui/button'

import { motion, AnimatePresence } from 'framer-motion'

export default function StoryCreator() {
  const router = useRouter()
  const bookRef = useRef<HTMLDivElement>(null)
  const [showBook, setShowBook] = useState(false)
  const [story, setStory] = useState('')

  const handleSubmit = (storyText: string) => {
    console.log("Pages: ", storyText)
    setStory(storyText)
    setShowBook(true)
    setTimeout(() => {
      bookRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }
  const navigateToQuiz = (storyText: string) => {
    router.push(`/quiz?story=${encodeURIComponent(storyText.response)}`)
  }
  return (
    <div className="relative min-h-screen overflow-hidden bg-blue-200">
      <Background />
      <Sidebar />
      <main className="relative z-10 p-8 ml-10">
        <h1 className="text-5xl font-bold text-blue-800 mb-8 font-serif">Story Creator</h1>
        <div className="flex flex-col items-center space-y-8">
          <div className="bg-white p-6 rounded-xl shadow-xl">
            <h2 className="text-2xl font-bold text-blue-600 mb-4">Draw Your Story</h2>
            <DrawableCanvas />
          </div>
          <div className="w-full max-w-2xl">
            <h2 className="text-2xl font-bold text-blue-600 mb-4">Tell Your Story</h2>
            <StoryInput onSubmit={handleSubmit} />
          </div>
        </div>
        <AnimatePresence>
          {showBook && (
            <motion.div 
              ref={bookRef}
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 50 }}
              transition={{ duration: 0.5 }}
              className="mt-16"
            >
              <h2 className="text-3xl font-bold text-blue-800 mb-8 font-serif text-center">Your Storybook</h2>
              {/* Pass the story data to the Book component */}
              <Book storyData={story} />
              <div className="mt-8 flex justify-center">
                <Button
                  onClick={() => navigateToQuiz(story)}
                  className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-full text-xl"
                >
                  Take a Quiz!
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}