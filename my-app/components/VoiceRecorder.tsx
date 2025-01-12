import React, { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Button } from "@/components/ui/button"
import { Mic, Square } from 'lucide-react'

interface VoiceRecorderProps {
  onTranscript: (transcript: string) => void;
}

export default function VoiceRecorder({ onTranscript }: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const recognitionRef = useRef<SpeechRecognition | null>(null)

  useEffect(() => {
    if (typeof window !== 'undefined' && 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      recognitionRef.current.continuous = true
      recognitionRef.current.interimResults = true

      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('')
        onTranscript(transcript)
      }
    }
  }, [onTranscript])

  const startRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.start()
      setIsRecording(true)
    }
  }

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      setIsRecording(false)
    }
  }

  return (
    <div className="mt-4 flex justify-center">
      <motion.div
        animate={{ scale: isRecording ? 1.1 : 1 }}
        transition={{ duration: 0.3 }}
      >
        <motion.button
          onClick={isRecording ? stopRecording : startRecording}
          className={`rounded-full p-4 ${
            isRecording ? 'bg-green-400 hover:bg-green-500' : 'bg-blue-400 hover:bg-blue-500'
          } font-comic text-2xl text-white shadow-lg`}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          animate={isRecording ? { scale: [1, 1.2, 1], rotate: [0, 10, -10, 0] } : {}}
          transition={{ duration: 0.5, repeat: isRecording ? Infinity : 0 }}
        >
          {isRecording ? <Square size={24} /> : <Mic size={24} />}
        </motion.button>
      </motion.div>
    </div>
  )
}

