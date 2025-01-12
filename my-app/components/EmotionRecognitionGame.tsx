'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import confetti from 'canvas-confetti'
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { CheckCircle, XCircle } from 'lucide-react'

const emotions = [
  { name: 'happy', emoji: '😊', bg: 'bg-yellow-400' },
  { name: 'sad', emoji: '😢', bg: 'bg-blue-400' },
  { name: 'angry', emoji: '😠', bg: 'bg-red-400' },
  { name: 'surprised', emoji: '😲', bg: 'bg-purple-400' },
]

interface EmotionItem {
  id: number
  emotion: typeof emotions[0]
}

export default function EmotionRecognitionGame() {
  const [items, setItems] = useState<EmotionItem[]>([])
  const [correctAnswers, setCorrectAnswers] = useState(0)
  const [incorrectAnswers, setIncorrectAnswers] = useState(0)
  const [score, setScore] = useState(0)
  const [gameOver, setGameOver] = useState(false)
  const [feedback, setFeedback] = useState<{ message: string, isCorrect: boolean } | null>(null)
  const [selectedItem, setSelectedItem] = useState<EmotionItem | null>(null)
  const [match, setMatch] = useState(0)

  useEffect(() => {
    startNewGame()
  }, [])

  useEffect(() => {
    if (gameOver) {
      saveMatchData()
    }
  }, [gameOver])



  const startNewGame = () => {
    const gameEmotions = emotions.slice(0, 4)
    const newItems = gameEmotions.map((emotion, index) => ({ id: index, emotion }))
    setItems(newItems)
    setCorrectAnswers(0)
    setIncorrectAnswers(0)
    setScore(0)
    setGameOver(false)
    setFeedback(null)
    setSelectedItem(null)
  }

  const handleItemClick = (item: EmotionItem) => {
    setSelectedItem(item)
  }

  const handleSlotClick = (targetEmotion: typeof emotions[0]) => {
    if (!selectedItem) return

    if (selectedItem.emotion.name === targetEmotion.name) {
      setItems(items.filter(i => i.id !== selectedItem.id))
      setCorrectAnswers(correctAnswers + 1)
      setScore(score + 1)
      setFeedback({ message: "Correct! Great job!", isCorrect: true })
      if (items.length === 1) {
        setGameOver(true)
        
        triggerConfetti()
      }
    } else {
      setIncorrectAnswers(incorrectAnswers + 1)
      setScore(score - 1)
      setFeedback({ message: "Oops! Try again!", isCorrect: false })
      shakeAnimation()
    }
    setSelectedItem(null)
  }

  const getTodayKey = () => {
    const today = new Date()
    const year = today.getFullYear()
    const month = String(today.getMonth() + 1).padStart(2, '0')
    const day = String(today.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  }

  const saveMatchData = () => {
    const todayKey = getTodayKey()
    const currentData = JSON.parse(localStorage.getItem('emotionGameScores') || '[]')

    const matchScore = correctAnswers - incorrectAnswers
    const averageScore = correctAnswers + incorrectAnswers > 0 ? matchScore / (correctAnswers) : 0

    const newMatch = {
      match: currentData.length > 0 ? currentData[currentData.length - 1].matches.length + 1 : 1,
      score: matchScore,
      correct: correctAnswers,
      incorrect: incorrectAnswers,
      totalQuestions: 5,
      averageScore: averageScore.toFixed(2),
    }

    if (currentData.length === 0 || currentData[currentData.length - 1].date !== todayKey) {
      currentData.push({
        date: todayKey,
        TotalMatches: 1,
        TotalAverageScore: newMatch.averageScore,
        matches: [newMatch]
      })
    } else {
      currentData[currentData.length - 1].matches.push(newMatch)
      currentData[currentData.length - 1].TotalMatches = currentData[currentData.length - 1].matches.length
      const totalAverageScore = (
        currentData[currentData.length - 1].matches.reduce((sum, match) => sum + parseFloat(match.averageScore), 0) /
        currentData[currentData.length - 1].matches.length
      ).toFixed(2)
      currentData[currentData.length - 1].TotalAverageScore = totalAverageScore
    }

    localStorage.setItem('emotionGameScores', JSON.stringify(currentData))
  }

  const triggerConfetti = () => {
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 }
    })
  }

  const shakeAnimation = () => {
    const gameArea = document.getElementById('emotion-game-area')
    if (gameArea) {
      gameArea.classList.add('shake')
      setTimeout(() => gameArea.classList.remove('shake'), 500)
    }
  }

  return (
    <div className="text-center">
      <h2 className="text-3xl font-bold mb-4 text-blue-600">Emotion Recognition Game</h2>
      <div className="flex justify-between items-center mb-4">
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline">How to Play</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>How to Play Emotion Recognition Game</DialogTitle>
              <DialogDescription>
                1. Click on an emoji to select it.<br />
                2. Then click on the matching emotion word.<br />
                3. Match all emotions correctly to win the game.<br />
                4. If you make a mistake, you can try again!<br />
                Have fun and learn to recognize emotions!
              </DialogDescription>
            </DialogHeader>
          </DialogContent>
        </Dialog>
      </div>
      <div id="emotion-game-area" className="mb-8">
        <div className="flex justify-center flex-wrap gap-4 mb-8">
          {emotions.map((emotion, index) => (
            <motion.div
              key={index}
              className={`w-32 h-32 rounded-lg border-4 border-white/50 shadow-lg ${emotion.bg} flex items-center justify-center text-xl font-bold text-white cursor-pointer`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleSlotClick(emotion)}
            >
              {emotion.name}
            </motion.div>
          ))}
        </div>
        <div className="flex justify-center flex-wrap gap-4">
          <AnimatePresence>
            {items.map((item) => (
              <motion.div
                key={item.id}
                className={`w-32 h-32 rounded-full shadow-lg ${item.emotion.bg} flex items-center justify-center text-4xl cursor-pointer ${selectedItem?.id === item.id ? 'border-4 border-blue-500' : ''}`}
                onClick={() => handleItemClick(item)}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {item.emotion.emoji}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
      <AnimatePresence>
        {feedback && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className={`text-xl font-bold mb-4 ${feedback.isCorrect ? 'text-green-500' : 'text-red-500'}`}
          >
            {feedback.message}
            {feedback.isCorrect ? <CheckCircle className="inline ml-2" /> : <XCircle className="inline ml-2" />}
          </motion.div>
        )}
      </AnimatePresence>
      {gameOver && (
        <motion.div
          className="mt-8 text-2xl font-bold text-green-500"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          🎉 Congratulations! You matched all the emotions! 🎉
        </motion.div>
      )}
      <motion.button
        onClick={startNewGame}
        className="mt-8 bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-6 rounded-full text-lg shadow-lg hover:shadow-xl transition-all duration-300"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        Restart Game
      </motion.button>
    </div>
  )
}