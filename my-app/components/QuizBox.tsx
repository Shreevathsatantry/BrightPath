import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'

interface QuizBoxProps {
  question: string
  options: string[]
  onAnswer: (selectedOption: string) => void
  isCorrect: boolean | null
}

export default function QuizBox({ question, options, onAnswer, isCorrect }: QuizBoxProps) {
  const [selectedOption, setSelectedOption] = useState<string | null>(null)

  const handleOptionClick = (option: string) => {
    setSelectedOption(option)
    onAnswer(option)
  }

  const getOptionStyle = (option: string) => {
    if (isCorrect === null) {
      return selectedOption === option ? 'bg-blue-500 text-white' : 'bg-purple-100 text-purple-800 hover:bg-purple-200'
    }
    if (isCorrect) {
      return selectedOption === option ? 'bg-green-500 text-white' : 'bg-purple-100 text-purple-800'
    }
    return selectedOption === option ? 'bg-red-500 text-white' : 'bg-purple-100 text-purple-800'
  }

  return (
    <motion.div
      className="bg-white p-8 rounded-xl shadow-2xl w-full flex flex-col justify-center"
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-3xl font-bold text-purple-800 mb-6 text-center">{question}</h2>
      <div className="grid grid-cols-2 gap-4">
        {options.map((option, index) => (
          <Button
            key={index}
            onClick={() => handleOptionClick(option)}
            disabled={isCorrect !== null}
            className={`p-4 text-xl font-semibold rounded-lg transition-colors ${getOptionStyle(option)}`}
          >
            {option}
          </Button>
        ))}
      </div>
      {isCorrect !== null && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className={`mt-4 text-center text-xl font-bold ${isCorrect ? 'text-green-500' : 'text-red-500'}`}
        >
          {isCorrect ? 'Correct!' : 'Incorrect. Try again!'}
        </motion.p>
      )}
    </motion.div>
  )
}

