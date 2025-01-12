'use client'

import { useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Camera, StopCircle } from 'lucide-react'

interface VideoSectionProps {
  isRecording: boolean
  onStart: () => void
  onStop: () => void
}

export default function VideoSection({ isRecording, onStart, onStop }: VideoSectionProps) {
  const videoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    if (isRecording) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream
          }
        })
        .catch(err => console.error('Error accessing camera:', err))
    } else if (videoRef.current) {
      const stream = videoRef.current.srcObject as MediaStream
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
      videoRef.current.srcObject = null
    }
  }, [isRecording])

  return (
    <div className="flex flex-col items-center p-6 space-y-6">
      <div className="w-full aspect-square relative overflow-hidden rounded-2xl bg-gradient-to-br from-purple-100 to-blue-200 border-4 border-purple-300">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 w-full h-full object-cover"
        />
        {!isRecording && !videoRef.current?.srcObject && (
          <motion.div 
            className="absolute inset-0 flex items-center justify-center"
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <div className="text-purple-400 w-24 h-24">
              <Camera size={96} />
            </div>
          </motion.div>
        )}
        {isRecording && (
          <motion.div 
            className="absolute top-4 right-4 w-6 h-6 rounded-full bg-red-500"
            animate={{ opacity: [1, 0] }}
            transition={{ duration: 1, repeat: Infinity }}
          />
        )}
      </div>
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={isRecording ? onStop : onStart}
        className={`w-48 h-14 rounded-full text-xl font-comic font-bold shadow-lg transition-all ${
          isRecording 
          ? 'bg-gradient-to-r from-red-400 to-pink-500 hover:from-red-500 hover:to-pink-600 text-white border-4 border-red-500' 
          : 'bg-gradient-to-r from-green-400 to-blue-500 hover:from-green-500 hover:to-blue-600 text-white border-4 border-green-500'
        }`}
      >
        {isRecording ? (
          <span className="flex items-center justify-center">
            <StopCircle size={24} className="mr-2" />
            Stop
          </span>
        ) : (
          <span className="flex items-center justify-center">
            <Camera size={24} className="mr-2" />
            Start
          </span>
        )}
      </motion.button>
    </div>
  )
}