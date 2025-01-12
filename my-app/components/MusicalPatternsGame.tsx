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

const instruments = ['🎹', '🎸', '🥁', '🎺']
const levels = [
  { name: 'Level 1', patternLength: 3, speed: 1000 },
  { name: 'Level 2', patternLength: 5, speed: 800 },
  { name: 'Level 3', patternLength: 7, speed: 600 },
]

interface GameMatch {
  match: number
  score: number
  correct: number
  incorrect: number
  totalQuestions: number
  averageScore: number
}

interface DayRecord {
  date: string
  TotalMatches: number
  TotalAverageScore: number
  matches: GameMatch[]
}

export default function MusicalPatternsGame() {
  const [currentLevel, setCurrentLevel] = useState(0)
  const [pattern, setPattern] = useState<string[]>([])
  const [playerPattern, setPlayerPattern] = useState<string[]>([])
  const [isPlaying, setIsPlaying] = useState(false)
  const [gameOver, setGameOver] = useState(false)
  const [feedback, setFeedback] = useState<string | null>(null)
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null)
  const [timeRemaining, setTimeRemaining] = useState<number>(30)
  const [timerActive, setTimerActive] = useState<boolean>(false)
  const [gameData, setGameData] = useState(() => {
    const today = new Date().toISOString().split('T')[0]
    const storedData = localStorage.getItem('musicalGameScore')
    const parsedData: DayRecord[] = storedData ? JSON.parse(storedData) : []
    
    // Find today's record or create a new one
    const todayRecord = parsedData.find(record => record.date === today)
    if (!todayRecord) {
      return {
        correct: 0,
        incorrect: 0,
        score: 0,
        match: 1,
        totalQuestions: 0
      }
    }
    
    const lastMatch = todayRecord.matches[todayRecord.matches.length - 1]
    return {
      correct: lastMatch.correct,
      incorrect: lastMatch.incorrect,
      score: lastMatch.score,
      match: lastMatch.match,
      totalQuestions: lastMatch.totalQuestions
    }
  })

  useEffect(() => {
    generatePattern()
  }, [currentLevel])

  useEffect(() => {
    if (timerActive && timeRemaining > 0) {
      const timer = setInterval(() => {
        setTimeRemaining(prev => prev - 1)
      }, 1000)
      return () => clearInterval(timer)
    } else if (timeRemaining === 0) {
      handleGameOver()
    }
  }, [timerActive, timeRemaining])

  const updateGameScores = (correct: boolean) => {
    const today = new Date().toISOString().split('T')[0]
    const storedData = localStorage.getItem('musicalGameScore')
    let gameScores: DayRecord[] = storedData ? JSON.parse(storedData) : []
    
    // Find or create today's record
    let todayRecord = gameScores.find(record => record.date === today)
    if (!todayRecord) {
      todayRecord = {
        date: today,
        TotalMatches: 0,
        TotalAverageScore: 0,
        matches: []
      }
      gameScores.push(todayRecord)
    }

    // Update current match data
    const newGameData = {
      ...gameData,
      correct: correct ? gameData.correct + 1 : gameData.correct,
      incorrect: correct ? gameData.incorrect : gameData.incorrect + 1,
      score: gameData.score + (correct ? 1 : 0),
      totalQuestions: gameData.totalQuestions + 1
    }

    // Create new match record
    const matchRecord: GameMatch = {
      match: gameData.match,
      score: newGameData.score,
      correct: newGameData.correct,
      incorrect: newGameData.incorrect,
      totalQuestions: newGameData.totalQuestions,
      averageScore: (newGameData.score) / newGameData.totalQuestions
    }

    // Update or add match to today's records
    const matchIndex = todayRecord.matches.findIndex(m => m.match === gameData.match)
    if (matchIndex >= 0) {
      todayRecord.matches[matchIndex] = matchRecord
    } else {
      todayRecord.matches.push(matchRecord)
    }

    // Update daily totals
    todayRecord.TotalMatches = todayRecord.matches.length
    todayRecord.TotalAverageScore = todayRecord.matches.reduce((sum, match) => sum + match.averageScore, 0) / todayRecord.TotalMatches

    // Sort matches by match number
    todayRecord.matches.sort((a, b) => a.match - b.match)

    // Sort records by date (newest first)
    gameScores.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())

    // Save updated records
    localStorage.setItem('musicalGameScore', JSON.stringify(gameScores))
    setGameData(newGameData)
  }

  const generatePattern = () => {
    setTimeRemaining(30)
    const newPattern = Array.from({ length: levels[currentLevel].patternLength }, () =>
      instruments[Math.floor(Math.random() * instruments.length)]
    )
    setPattern(newPattern)
    setPlayerPattern([])
    setIsPlaying(true)
    playPattern(newPattern)
    setTimerActive(true)
  }

  const playPattern = async (patternToPlay: string[]) => {
    for (let instrument of patternToPlay) {
      setFeedback(instrument)
      await new Promise(resolve => setTimeout(resolve, levels[currentLevel].speed))
      setFeedback(null)
      await new Promise(resolve => setTimeout(resolve, 200))
    }
    setIsPlaying(false)
  }

  const handleInstrumentClick = (instrument: string) => {
    if (isPlaying || timeRemaining === 0) return

    const newPlayerPattern = [...playerPattern, instrument]
    setPlayerPattern(newPlayerPattern)

    if (newPlayerPattern.length === pattern.length) {
      const correct = newPlayerPattern.every((instrument, index) => instrument === pattern[index])
      setIsCorrect(correct)
      updateGameScores(correct)

      if (correct) {
        setFeedback('Correct! Great job!')
        if (gameData.score + 1 === 3 && currentLevel < levels.length - 1) {
          setCurrentLevel(currentLevel + 1)
          setGameData(prev => ({ 
            ...prev, 
            score: 0,
            match: prev.match + 1 
          }))
        } else if (gameData.score + 1 === 3 && currentLevel === levels.length - 1) {
          setGameOver(true)
          triggerConfetti()
        } else {
          setTimeout(generatePattern, 1500)
        }
      } else {
        setFeedback('Oops! Try again!')
        setTimeout(() => {
          setIsCorrect(null)
          generatePattern()
        }, 1500)
      }
    }
  }

  const handleGameOver = () => {
    setGameOver(true)
    setTimerActive(false)
    triggerConfetti()
  }

  const triggerConfetti = () => {
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 }
    })
  }

  return (
    <div className="text-center">
      <h2 className="text-3xl font-bold mb-4 text-blue-600">Musical Patterns Game</h2>
      <div className="flex justify-between items-center mb-4">
        <p className="text-xl">{levels[currentLevel].name}</p>
        <p className="text-xl">Score: {gameData.score}/3</p>
        <p className="text-xl">Correct: {gameData.correct}</p>
        <p className="text-xl">Incorrect: {gameData.incorrect}</p>
        <p className="text-xl">Match: {gameData.match}</p>
        <p className="text-xl">Time: {timeRemaining}s</p>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline">How to Play</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>How to Play Musical Patterns Game</DialogTitle>
              <DialogDescription>
                1. Watch the pattern of instruments.<br/>
                2. Click the instruments in the same order to repeat the pattern.<br/>
                3. Complete 3 patterns correctly to advance to the next level.<br/>
                4. There are 3 levels with increasing difficulty.<br/>
                5. Win the game by completing all levels!<br/>
                Have fun and improve your memory and pattern recognition skills!
              </DialogDescription>
            </DialogHeader>
          </DialogContent>
        </Dialog>
      </div>
      <div className="mb-8">
        <div className="flex justify-center gap-4 mb-4">
          {instruments.map((instrument, index) => (
            <motion.button
              key={index}
              className="w-40 h-40 text-6xl bg-purple-200 rounded-lg shadow-lg flex items-center justify-center"
              onClick={() => handleInstrumentClick(instrument)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              disabled={isPlaying || timeRemaining === 0}
            >
              {instrument}
            </motion.button>
          ))}
        </div>
        <AnimatePresence>
          {feedback && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="text-2xl font-bold mb-4"
            >
              {feedback}
              {isCorrect !== null && (
                <span className="ml-2">
                  {isCorrect ? <CheckCircle className="inline text-green-500" /> : <XCircle className="inline text-red-500" />}
                </span>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      {gameOver ? (
        <motion.div
          className="text-2xl font-bold text-green-500"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          🎉 Congratulations! You've completed all levels! 🎉
        </motion.div>
      ) : (
        <Button 
          onClick={generatePattern} 
          disabled={isPlaying || timeRemaining === 0}
          className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
        >
          {gameData.score === 0 ? 'Start Game' : 'Next Pattern'}
        </Button>
      )}
    </div>
  )
}