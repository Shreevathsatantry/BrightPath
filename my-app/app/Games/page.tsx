// Add "use client" at the top to make this a Client Component
'use client';

import { useEffect, useState } from 'react';
import Image from 'next/image'
import Link from 'next/link'
import Sidebar from '@/components/Sidebar';

const games = [
  {
    id: 'color-matching',
    title: 'Color Matching Game',
    description: 'Match colors by clicking on shapes',
    color: 'bg-yellow-300',
    icon: '🎨'
  },
  {
    id: 'emotion-recognition',
    title: 'Emotion Recognition',
    description: 'Match facial expressions with emotions to enhance social understanding',
    color: 'bg-pink-300',
    icon: '😊'
  },
  {
    id: 'shape-sorting',
    title: 'Shape Sorting Adventure',
    description: 'Sort geometric shapes into their matching outlines',
    color: 'bg-green-300',
    icon: '⭐'
  },
  {
    id: 'musical-patterns',
    title: 'Musical Patterns',
    description: 'Repeat musical sequences using virtual instruments',
    color: 'bg-purple-300',
    icon: '🎵'
  },
  {
    id: 'memory-card',
    title: 'Memory Card Game',
    description: 'Match pairs of cards with similar images to build memory skills',
    color: 'bg-red-300',
    icon: '🃏'
  },
  // {
  //   id: 'word-formation',
  //   title: 'Word Formation Challenge',
  //   description: 'Create words from jumbled letters',
  //   color: 'bg-indigo-300',
  //   icon: '📚'
  // },
  {
    id: 'grammar-detective',
    title: 'Grammar Detective',
    description: 'Identify correct and incorrect grammar in sentences',
    color: 'bg-teal-300',
    icon: '🕵️'
  },
  {
    id: 'basic-arithmetic',
    title: 'Basic Arithmetic Adventure',
    description: 'Solve addition, subtraction, multiplication, and division problems',
    color: 'bg-amber-300',
    icon: '🔢'
  },
  {
    id: 'shape-pattern',
    title: 'Shape & Pattern Puzzle',
    description: 'Identify and complete patterns with shapes',
    color: 'bg-lime-300',
    icon: '🔷'
  },
]

export default function Home() {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    // Set state to true only on the client-side
    setIsClient(true);
  }, []);

  if (!isClient) return null; // Render nothing until mounted on the client side

  const { motion } = require('framer-motion'); // Import motion dynamically inside useEffect

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <Sidebar />
          <motion.div
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-4xl md:text-5xl font-bold text-blue-600 mb-4">
              BrightPath Games
            </h1>
            <p className="text-xl text-blue-600/80">
              Guiding children with autism to learn, grow, and shine!
            </p>
          </motion.div>
        </div>

        {/* Games Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {games.map((game, index) => (
            <motion.div
              key={game.id}
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: index * 0.1 }}
            >
              <Link href={`Games/games/${game.id}`}>

                <div className={`game-card relative overflow-hidden ${game.color} rounded-2xl p-6 shadow-lg hover:shadow-xl cursor-pointer h-64`}>
                  <div className="text-4xl mb-4">{game.icon}</div>
                  <h2 className="text-2xl font-bold text-gray-800 mb-2">
                    {game.title}
                  </h2>
                  <p className="text-gray-700">
                    {game.description}
                  </p>
                  <div className="mt-4">
                    <span className="inline-flex items-center justify-center px-4 py-2 bg-white/30 backdrop-blur-sm rounded-full text-sm font-medium text-gray-800">
                      Play Now →
                    </span>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>

        {/* Mascot */}
        <motion.div
          className="fixed bottom-8 right-8"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.5, type: "spring" }}
        >
          <div className="w-24 h-24 bg-yellow-400 rounded-full flex items-center justify-center text-4xl shadow-lg">
            🐕
          </div>
        </motion.div>
      </div>
    </main>
  );
}
