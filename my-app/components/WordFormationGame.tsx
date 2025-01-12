// 'use client'

// import { useState, useEffect } from 'react'
// import { motion, AnimatePresence } from 'framer-motion'
// import confetti from 'canvas-confetti'
// import { Button } from "@/components/ui/button"
// import {
//   Dialog,
//   DialogContent,
//   DialogDescription,
//   DialogHeader,
//   DialogTitle,
//   DialogTrigger,
// } from "@/components/ui/dialog"

// const levels = [
//   { name: 'Level 1', words: ['CAT', 'DOG', 'BIRD', 'FISH', 'TREE'], time: 60 },
//   { name: 'Level 2', words: ['HOUSE', 'TREE', 'FLOWER', 'PARK', 'STONE'], time: 90 },
//   { name: 'Level 3', words: ['ELEPHANT', 'GIRAFFE', 'PENGUIN', 'CATER', 'HORSE'], time: 120 },
// ]

// export default function WordFormationGame() {
//   const [currentLevel, setCurrentLevel] = useState(0)
//   const [currentWord, setCurrentWord] = useState('')
//   const [jumbledWord, setJumbledWord] = useState('')
//   const [userWord, setUserWord] = useState('')
//   const [score, setScore] = useState(0)
//   const [timeLeft, setTimeLeft] = useState(levels[currentLevel].time)
//   const [gameOver, setGameOver] = useState(false)
//   const [feedback, setFeedback] = useState<string | null>(null)

//   useEffect(() => {
//     if (timeLeft > 0 && !gameOver) {
//       const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000)
//       return () => clearTimeout(timer)
//     } else if (timeLeft === 0 && !gameOver) {
//       endGame()
//     }
//   }, [timeLeft, gameOver])

//   useEffect(() => {
//     startNewWord()
//   }, [currentLevel])

//   const startNewWord = () => {
//     // Filter words of 3 and 4 letters from the current level
//     const levelWords = levels[currentLevel].words.filter(word => word.length === 3 || word.length === 4);
    
//     const newWord = levelWords[Math.floor(Math.random() * levelWords.length)];
//     setCurrentWord(newWord)
//     setJumbledWord(jumbleWord(newWord))
//     setUserWord('')
//     setFeedback(null)
//   }

//   const jumbleWord = (word: string) => {
//     return word.split('').sort(() => Math.random() - 0.5).join('')
//   }

//   const handleLetterClick = (letter: string) => {
//     setUserWord(userWord + letter)
//     setJumbledWord(jumbledWord.replace(letter, ''))
//   }

//   const handleSubmit = () => {
//     if (userWord === currentWord) {
//       setScore(score + 1)
//       setFeedback('Correct!')
//       if (score + 1 === levels[currentLevel].words.length) {
//         if (currentLevel < levels.length - 1) {
//           setCurrentLevel(currentLevel + 1)
//           setScore(0)
//           setTimeLeft(levels[currentLevel + 1].time)
//         } else {
//           endGame()
//         }
//       } else {
//         // Move to the next word if the current word is correct
//         startNewWord()
//       }
//     } else {
//       setFeedback('Try again!')
//       setUserWord('')
//       setJumbledWord(jumbleWord(currentWord))
//     }
//   }

//   const endGame = () => {
//     setGameOver(true)
//     if (score === levels[currentLevel].words.length && currentLevel === levels.length - 1) {
//       triggerConfetti()
//     }
//   }

//   const triggerConfetti = () => {
//     confetti({
//       particleCount: 100,
//       spread: 70,
//       origin: { y: 0.6 }
//     })
//   }

//   return (
//     <div className="text-center">
//       <h2 className="text-3xl font-bold mb-4 text-blue-600">Word Formation Challenge</h2>
//       <div className="flex justify-between items-center mb-4">
//         <p className="text-xl">{levels[currentLevel].name}</p>
//         <p className="text-xl">Score: {score}/{levels[currentLevel].words.length}</p>
//         <p className="text-xl">Time: {timeLeft}s</p>
//         <Dialog>
//           <DialogTrigger asChild>
//             <Button variant="outline">How to Play</Button>
//           </DialogTrigger>
//           <DialogContent>
//             <DialogHeader>
//               <DialogTitle>How to Play Word Formation Challenge</DialogTitle>
//               <DialogDescription>
//                 1. Unscramble the jumbled letters to form the correct word.<br />
//                 2. Click on the letters to form your word.<br />
//                 3. Submit your answer when you think it's correct.<br />
//                 4. Complete all words in each level to advance.<br />
//                 5. There are 3 levels with increasing difficulty.<br />
//                 6. Win the game by completing all levels before time runs out!<br />
//                 Have fun and improve your spelling skills!
//               </DialogDescription>
//             </DialogHeader>
//           </DialogContent>
//         </Dialog>
//       </div>
//       {!gameOver ? (
//         <div className="mb-8">
//           <div className="text-2xl mb-4">Jumbled Word: {jumbledWord}</div>
//           <div className="text-2xl mb-4">Your Word: {userWord}</div>
//           <div className="flex justify-center gap-2 mb-4">
//             {jumbledWord.split('').map((letter, index) => (
//               <motion.button
//                 key={index}
//                 className="bg-blue-500 text-white px-4 py-2 rounded"
//                 onClick={() => handleLetterClick(letter)}
//                 whileHover={{ scale: 1.1 }}
//                 whileTap={{ scale: 0.9 }}
//               >
//                 {letter}
//               </motion.button>
//             ))}
//           </div>
//           <Button onClick={handleSubmit} className="bg-green-500 text-white px-4 py-2 rounded">
//             Submit
//           </Button>
//           {feedback && (
//             <div className={`mt-4 text-xl ${feedback === 'Correct!' ? 'text-green-500' : 'text-red-500'}`}>
//               {feedback}
//             </div>
//           )}
//         </div>
//       ) : (
//         <div>
//           <p className="text-xl mb-4">
//             Game Over! Your final score: {score}/{levels[currentLevel].words.length}
//           </p>
//           {score === levels[currentLevel].words.length && currentLevel === levels.length - 1 ? (
//             <motion.div
//               className="text-2xl font-bold text-green-500"
//               initial={{ opacity: 0, y: 20 }}
//               animate={{ opacity: 1, y: 0 }}
//             >
//               🎉 Congratulations! You've completed all levels! 🎉
//             </motion.div>
//           ) : (
//             <Button
//               onClick={() => {
//                 setCurrentLevel(0)
//                 setScore(0)
//                 setTimeLeft(levels[0].time)
//                 setGameOver(false)
//                 startNewWord()
//               }}
//               className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
//             >
//               Play Again
//             </Button>
//           )}
//         </div>
//       )}
//     </div>
//   )
// }
