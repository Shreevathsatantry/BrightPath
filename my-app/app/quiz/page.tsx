'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Background from '../../components/Background'
import Sidebar from '../../components/Sidebar'
import QuizBox from '../../components/QuizBox'
import Bot from '../../components/Bot'
import { Button } from '@/components/ui/button'

const questions = [
  {
    question: "What color is the sky on a clear day?",
    options: ["Blue", "Green", "Red", "Yellow"],
    correctAnswer: "Blue"
  },
  {
    question: "How many legs does a cat have?",
    options: ["Two", "Four", "Six", "Eight"],
    correctAnswer: "Four"
  },
  {
    question: "Which animal says 'moo'?",
    options: ["Dog", "Cat", "Cow", "Sheep"],
    correctAnswer: "Cow"
  },
  {
    question: "What shape is a ball?",
    options: ["Square", "Triangle", "Circle", "Rectangle"],
    correctAnswer: "Circle"
  },
  {
    question: "Which fruit is red and grows on a tree?",
    options: ["Banana", "Orange", "Apple", "Grapes"],
    correctAnswer: "Apple"
  }
]

export default function QuizPage() {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [quizStarted, setQuizStarted] = useState(false)
  const [score, setScore] = useState(0)
  const [lastAnswerCorrect, setLastAnswerCorrect] = useState<boolean | null>(null)

  const startQuiz = () => {
    setQuizStarted(true)
    setScore(0)
    setCurrentQuestion(0)
    setLastAnswerCorrect(null)
  }

  const handleAnswer = (selectedOption: string) => {
    const isCorrect = selectedOption === questions[currentQuestion].correctAnswer
    setLastAnswerCorrect(isCorrect)
    if (isCorrect) {
      setScore(score + 1)
    }
    setTimeout(() => {
      if (currentQuestion < questions.length - 1) {
        setCurrentQuestion(currentQuestion + 1)
        setLastAnswerCorrect(null)
      } else {
        // Quiz finished
        setQuizStarted(false)
      }
    }, 1500)
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-b from-purple-300 to-pink-200">
      <Background />
      <Sidebar />
      <main className="relative z-10 p-8 ml-10">
        <h1 className="text-5xl font-bold text-purple-800 mb-8 font-serif text-center">Fun Quiz Time!</h1>
        
        {!quizStarted ? (
          <motion.div 
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center space-y-4"
          >
            {currentQuestion > 0 && (
              <h2 className="text-3xl font-bold text-purple-600">Your Score: {score}/{questions.length}</h2>
            )}
            <Button onClick={startQuiz} className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-full text-xl">
              {currentQuestion > 0 ? 'Play Again' : 'Start Quiz'}
            </Button>
          </motion.div>
        ) : (
          <div className="flex flex-col items-center justify-center h-[calc(100vh-200px)]">
            <div className="mb-4">
              <Bot isCorrect={lastAnswerCorrect} />
            </div>
            <div className="w-full max-w-2xl">
              <QuizBox 
                question={questions[currentQuestion].question}
                options={questions[currentQuestion].options}
                onAnswer={handleAnswer}
                isCorrect={lastAnswerCorrect}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
