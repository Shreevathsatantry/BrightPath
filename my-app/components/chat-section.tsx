'use client'

import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Volume2, VolumeX, Pause, Play } from 'lucide-react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select_commu"

interface Message {
  type: 'user' | 'bot'
  content: string
}

interface ChatSectionProps {
  messages: Message[]
  currentSpeech: string
  isProcessing: boolean
  isRecording: boolean
}

interface VoiceOption {
  name: string
  label: string
  emoji: string
}

const voiceOptions: VoiceOption[] = [
  { name: 'girl', label: 'Girl', emoji: '👧' },
  { name: 'boy', label: 'Boy', emoji: '👦' },
  { name: 'woman', label: 'Woman', emoji: '👩' },
  { name: 'man', label: 'Man', emoji: '👨' },
]

export default function ChatSection({ messages, currentSpeech, isProcessing, isRecording }: ChatSectionProps) {
  const chatRef = useRef<HTMLDivElement>(null)
  const speechRef = useRef<SpeechSynthesisUtterance | null>(null)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([])
  const [selectedVoice, setSelectedVoice] = useState<string>('girl')

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight
    }
  }, [messages, currentSpeech])

  useEffect(() => {
    if (typeof window !== 'undefined') {
      speechRef.current = new SpeechSynthesisUtterance()
      speechRef.current.rate = 1
      speechRef.current.pitch = 1
      speechRef.current.volume = 1

      const loadVoices = () => {
        const availableVoices = window.speechSynthesis.getVoices()
        setVoices(availableVoices)
        setAppropriateVoice('girl')
      }

      if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = loadVoices
      }

      loadVoices()

      speechRef.current.onend = () => {
        setIsSpeaking(false)
        setIsPaused(false)
      }
    }

    return () => {
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel()
      }
    }
  }, [])

  useEffect(() => {
    const lastMessage = messages[messages.length - 1]
    if (lastMessage?.type === 'bot' && !isProcessing) {
      speakMessage(lastMessage.content)
    }
  }, [messages, isProcessing])

  const setAppropriateVoice = (voiceType: string) => {
    if (speechRef.current && voices.length > 0) {
      let voice: SpeechSynthesisVoice | undefined

      switch (voiceType) {
        case 'girl':
          voice = voices.find(v => v.name.includes('Girl') || v.name.includes('Female'))
          speechRef.current.pitch = 1.2
          break
        case 'boy':
          voice = voices.find(v => v.name.includes('Boy') || (v.name.includes('Male') && !v.name.includes('Female')))
          speechRef.current.pitch = 1.1
          break
        case 'woman':
          voice = voices.find(v => v.name.includes('Female') && !v.name.includes('Girl'))
          speechRef.current.pitch = 1
          break
        case 'man':
          voice = voices.find(v => v.name.includes('Male') && !v.name.includes('Boy'))
          speechRef.current.pitch = 0.9
          break
      }

      if (voice) {
        speechRef.current.voice = voice
      } else {
        // Fallback to a default voice if no matching voice is found
        speechRef.current.voice = voices[0]
      }
    }
  }

  const speakMessage = (text: string) => {
    if (speechRef.current && window.speechSynthesis) {
      window.speechSynthesis.cancel()
      speechRef.current.text = text
      setAppropriateVoice(selectedVoice)
      window.speechSynthesis.speak(speechRef.current)
      setIsSpeaking(true)
      setIsPaused(false)
    }
  }

  const toggleSpeech = () => {
    if (window.speechSynthesis) {
      if (isPaused) {
        window.speechSynthesis.resume()
        setIsPaused(false)
      } else {
        window.speechSynthesis.pause()
        setIsPaused(true)
      }
    }
  }

  const stopSpeech = () => {
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel()
      setIsSpeaking(false)
      setIsPaused(false)
    }
  }

  const handleVoiceChange = (value: string) => {
    setSelectedVoice(value)
    setAppropriateVoice(value)
  }

  return (
    <div className="h-[500px] flex flex-col p-6">
      <motion.div 
        className="mb-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Select onValueChange={handleVoiceChange} value={selectedVoice}>
          <SelectTrigger className="w-full bg-gradient-to-r from-yellow-300 to-orange-300 border-4 border-yellow-400 rounded-full text-lg font-comic font-bold text-purple-700 focus:ring-4 focus:ring-purple-300 focus:border-purple-400 transition-all duration-300">
            <SelectValue placeholder="Choose a voice" />
          </SelectTrigger>
          <SelectContent className="bg-gradient-to-r from-yellow-100 to-orange-100 border-4 border-yellow-400 rounded-2xl overflow-hidden">
            {voiceOptions.map((voice) => (
              <SelectItem 
                key={voice.name} 
                value={voice.name}
                className="text-lg font-comic font-semibold text-purple-700 hover:bg-yellow-200 focus:bg-yellow-300 focus:text-purple-800 transition-all duration-200"
              >
                <span className="mr-2">{voice.emoji}</span> {voice.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </motion.div>
      <div 
        ref={chatRef}
        className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-pink-400 scrollbar-track-pink-100"
      >
        <div className="space-y-4">
          <AnimatePresence>
            {messages.map((message, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl p-4 ${
                    message.type === 'user' 
                    ? 'bg-gradient-to-r from-blue-400 to-indigo-500 text-white border-4 border-blue-500' 
                    : 'bg-gradient-to-r from-purple-200 to-pink-300 text-purple-900 border-4 border-purple-300'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <p className="text-sm font-comic">{message.content}</p>
                    {message.type === 'bot' && (
                      <div className="flex gap-2 mt-1">
                        {isSpeaking && message === messages[messages.length - 1] && (
                          <motion.button
                            onClick={toggleSpeech}
                            className="p-1 rounded-full bg-purple-300 hover:bg-purple-400 transition-colors"
                            whileHover={{ scale: 1.1 }}
                            whileTap={{ scale: 0.9 }}
                          >
                            {isPaused ? (
                              <Play className="w-4 h-4 text-purple-900" />
                            ) : (
                              <Pause className="w-4 h-4 text-purple-900" />
                            )}
                          </motion.button>
                        )}
                        <motion.button
                          onClick={() => speakMessage(message.content)}
                          className="p-1 rounded-full bg-purple-300 hover:bg-purple-400 transition-colors"
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                        >
                          <Volume2 className="w-4 h-4 text-purple-900" />
                        </motion.button>
                      </div>
                    )}
                  </div>
                  {message.type === 'bot' && isSpeaking && message === messages[messages.length - 1] && (
                    <motion.div
                      className="mt-2 flex gap-1"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      {[...Array(3)].map((_, i) => (
                        <motion.div
                          key={i}
                          className="w-2 h-2 rounded-full bg-purple-400"
                          animate={{ scale: [1, 1.5, 1] }}
                          transition={{
                            duration: 1,
                            repeat: Infinity,
                            delay: i * 0.2
                          }}
                        />
                      ))}
                    </motion.div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {currentSpeech && isRecording && (
            <motion.div 
              className="flex justify-end"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="max-w-[80%] rounded-2xl p-4 bg-gradient-to-r from-blue-300 to-indigo-300 border-4 border-blue-400">
                <p className="text-sm font-comic text-blue-900">{currentSpeech}</p>
              </div>
            </motion.div>
          )}
          
          {isProcessing && (
            <motion.div 
              className="flex justify-start"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <div className="max-w-[80%] rounded-2xl p-4 bg-gradient-to-r from-green-200 to-teal-300 border-4 border-green-300">
                <div className="flex items-center space-x-2">
                  <motion.div 
                    className="h-4 w-4 bg-green-500 rounded-full"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                  />
                  <motion.div 
                    className="h-4 w-4 bg-green-500 rounded-full"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 1, delay: 0.2, repeat: Infinity }}
                  />
                  <motion.div 
                    className="h-4 w-4 bg-green-500 rounded-full"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 1, delay: 0.4, repeat: Infinity }}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}