'use client'


import React from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import Background from '../components/Background' // Ensure the relative path is correct
import Sidebar from '../components/Sidebar'
import FeatureCard from '../components/FeatureCard'
import AnimatedCharacter from '../components/AnimatedCharacter'
import HeroSection from '../components/HeroSection'
import FloatingAnimation from '../components/FloatingAnimation'


export default function LandingPage() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-blue-200">
      <Background />
      <Sidebar />
      <main className="relative z-10 p-8 ml-10">
        <HeroSection />
        <div className="grid grid-cols-1 gap-8 mt-16 md:grid-cols-2 lg:grid-cols-4">
          <FeatureCard
            title="BrightPath Games"
            description="Fun and interactive games to boost cognitive skills"
            icon="🎮"
            color="bg-yellow-300"
          />
          <FeatureCard
            title="Progress Tracker"
            description="Monitor your child's learning journey"
            icon="📊"
            color="bg-green-300"
          />
          <FeatureCard
            title="BrightPath Stories"
            description="Engaging tales that teach and entertain"
            icon="📚"
            color="bg-pink-300"
          />
          <FeatureCard
            title="Social Skills Builder"
            description="Learn to navigate social situations with confidence"
            icon="🤝"
            color="bg-purple-300"
          />
        </div>
        {/* <Link href="/story-creator" passHref>
          <motion.button
            className="mt-8 bg-blue-500 text-white px-6 py-3 rounded-full text-xl font-bold"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Try Story Creator
          </motion.button>
        </Link> */}
        <AnimatedCharacter character="🐶" style={{ top: '20%', left: '10%' }} />
        <AnimatedCharacter character="🐱" style={{ top: '60%', right: '15%' }} />
        <AnimatedCharacter character="🐰" style={{ top: '40%', left: '80%' }} />
        <FloatingAnimation animation="float">
          <span className="text-4xl">🎈</span>
        </FloatingAnimation>
        <FloatingAnimation animation="twinkle">
          <span className="text-4xl">⭐</span>
        </FloatingAnimation>
      </main>
    </div>
  )
}

