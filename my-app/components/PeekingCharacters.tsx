import React from 'react'
import { motion } from 'framer-motion'

export default function PeekingCharacters() {
  return (
    <div className="absolute -top-32 left-0 right-0 flex justify-center pointer-events-none">
      <motion.div
        className="relative"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.5, type: 'spring', bounce: 0.4 }}
      >
        {/* Dog */}
        <motion.div
          className="absolute -left-20 top-0"
          animate={{ y: [0, -10, 0] }}
          transition={{ duration: 4, repeat: Infinity, repeatType: 'reverse' }}
        >
          <div className="relative w-48 h-48">
            <div className="absolute bottom-0 w-full">
              {/* Dog's head */}
              <div className="relative w-32 h-32 bg-[#8B4513] rounded-full">
                <div className="absolute w-24 h-20 bg-white bottom-0 left-4 rounded-b-full" /> {/* Muzzle */}
                <div className="absolute w-8 h-12 bg-[#8B4513] -left-4 top-2 rounded-l-full transform rotate-45" /> {/* Left ear */}
                <div className="absolute w-8 h-12 bg-[#8B4513] right-0 top-2 rounded-r-full transform -rotate-45" /> {/* Right ear */}
                <div className="absolute w-3 h-3 bg-black rounded-full left-8 top-8" /> {/* Left eye */}
                <div className="absolute w-3 h-3 bg-black rounded-full right-8 top-8" /> {/* Right eye */}
                <div className="absolute w-8 h-4 bg-[#FF69B4] bottom-6 left-12 rounded-full" /> {/* Tongue */}
              </div>
              {/* Dog's collar */}
              <div className="absolute bottom-0 left-8 w-16 h-4 bg-red-500 rounded-full">
                <div className="absolute w-4 h-4 bg-yellow-400 rounded-full left-6 -bottom-1" /> {/* Tag */}
              </div>
              {/* Dog's paws */}
              <div className="absolute bottom-0 left-4 w-8 h-12 bg-[#8B4513] rounded-b-lg" />
              <div className="absolute bottom-0 right-4 w-8 h-12 bg-[#8B4513] rounded-b-lg" />
            </div>
          </div>
        </motion.div>

        {/* Cat */}
        <motion.div
          className="absolute -right-16 top-4"
          animate={{ y: [0, -8, 0] }}
          transition={{ duration: 3, repeat: Infinity, repeatType: 'reverse', delay: 1 }}
        >
          <div className="relative w-40 h-40">
            <div className="absolute bottom-0 w-full">
              {/* Cat's head */}
              <div className="relative w-28 h-28 bg-[#FFA500] rounded-full">
                <div className="absolute w-20 h-16 bg-white bottom-0 left-4 rounded-b-full" /> {/* Muzzle */}
                <div className="absolute w-8 h-12 bg-[#FFA500] -left-2 -top-2 rounded-t-full transform rotate-15" /> {/* Left ear */}
                <div className="absolute w-8 h-12 bg-[#FFA500] right-0 -top-2 rounded-t-full transform -rotate-15" /> {/* Right ear */}
                <div className="absolute w-2.5 h-2.5 bg-black rounded-full left-8 top-8" /> {/* Left eye */}
                <div className="absolute w-2.5 h-2.5 bg-black rounded-full right-8 top-8" /> {/* Right eye */}
                <div className="absolute w-6 h-3 bg-[#FF69B4] bottom-5 left-11 rounded-full" /> {/* Tongue */}
              </div>
              {/* Cat's collar */}
              <div className="absolute bottom-0 left-6 w-16 h-3 bg-green-500 rounded-full">
                <div className="absolute w-3 h-3 bg-yellow-400 rounded-full left-6 -bottom-1" /> {/* Bell */}
              </div>
              {/* Cat's paws */}
              <div className="absolute bottom-0 left-4 w-6 h-10 bg-[#FFA500] rounded-b-lg" />
              <div className="absolute bottom-0 right-4 w-6 h-10 bg-[#FFA500] rounded-b-lg" />
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}

