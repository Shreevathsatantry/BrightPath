'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import VideoSection from '../../components/video-section'
import ChatSection from '../../components/chat-section'
import Sidebar from '../../components/Sidebar'
import { Sun, Cloud, Star, TreesIcon as Tree } from 'lucide-react'

export default function VideoChat() {
  const [isRecording, setIsRecording] = useState(false)
  const [messages, setMessages] = useState([])
  const [currentSpeech, setCurrentSpeech] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [botContent, setBotContent] = useState('')
  const mediaRecorderRef = useRef(null)
  const recognitionRef = useRef(null)
  const chunksRef = useRef([])
  const lastMessageRef = useRef('')

  // Initialize speech recognition
  useEffect(() => {
    if (typeof window !== 'undefined' && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
      recognitionRef.current = new (window.SpeechRecognition || window.webkitSpeechRecognition)()
      recognitionRef.current.continuous = true
      recognitionRef.current.interimResults = true

      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('')
        setCurrentSpeech(transcript)
      }

      recognitionRef.current.onend = () => {
        if (isRecording) {
          recognitionRef.current.start()
        }
      }
    }
  }, [isRecording]) // Ensures re-initialization on dependency changes

  const fetchVideoAnalysis = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/VideoAnalyzer", {
        method: "GET",
      });
      const data = await response.json();
      console.log('Video analysis result:', data);
      return data.response; // Return the response
    } catch (error) {
      console.error('Error communicating with the server:', error);
      return "Sorry, I couldn't analyze the video.";
    }
  };
  

  const startRecording = async () => {
    try {
      if (recognitionRef.current) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        const mediaRecorder = new MediaRecorder(stream)
        chunksRef.current = []

        mediaRecorder.ondataavailable = (e) => {
          chunksRef.current.push(e.data)
        }

        mediaRecorder.onstop = async () => {
          const blob = new Blob(chunksRef.current, { type: 'video/webm' })
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `Video.mp4`
          document.body.appendChild(a)
          a.click()
          document.body.removeChild(a)
          URL.revokeObjectURL(url)
        }

        mediaRecorderRef.current = mediaRecorder
        mediaRecorder.start()
        recognitionRef.current.start()
        setIsRecording(true)
        setCurrentSpeech('')
      } else {
        console.error('SpeechRecognition API is not available.')
      }
    } catch (err) {
      console.error('Error accessing media devices:', err)
    }
  }

  const stopRecording = async () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      recognitionRef.current.stop();
      setIsRecording(false);
  
      if (currentSpeech && currentSpeech !== lastMessageRef.current) {
        lastMessageRef.current = currentSpeech;
        setMessages((prev) => [
          ...prev,
          {
            type: "user",
            content: currentSpeech,
          },
        ]);
  
        setIsProcessing(true);
        const botResponse = await fetchVideoAnalysis(); // Fetch bot response
        setMessages((prev) => [
          ...prev,
          {
            type: "bot",
            content: botResponse, // Use the fetched response directly
          },
        ]);
        setIsProcessing(false);
      }
      setCurrentSpeech("");
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-b from-sky-300 via-sky-200 to-green-200">
      {/* Decorative elements */}
      <Sidebar />
      <div className="absolute inset-0 pointer-events-none">
        <motion.div
          className="absolute top-10 left-10 text-yellow-400"
          animate={{ 
            rotate: 360,
            scale: [1, 1.2, 1],
          }}
          transition={{ 
            rotate: { duration: 20, repeat: Infinity, ease: "linear" },
            scale: { duration: 2, repeat: Infinity, repeatType: "reverse" }
          }}
        >
          <Sun size={60} />
        </motion.div>

        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute"
            initial={{ x: Math.random() * 100, y: Math.random() * 100 }}
            animate={{
              x: [Math.random() * window.innerWidth, Math.random() * window.innerWidth],
              y: [Math.random() * window.innerHeight, Math.random() * window.innerHeight],
            }}
            transition={{
              duration: Math.random() * 20 + 10,
              repeat: Infinity,
              ease: "linear"
            }}
          >
            <Star className="text-yellow-300" size={24} />
          </motion.div>
        ))}

        {/* Clouds */}
        {[...Array(3)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute"
            initial={{ x: -100, y: Math.random() * 200 }}
            animate={{ x: window.innerWidth + 100 }}
            transition={{
              duration: Math.random() * 60 + 30,
              repeat: Infinity,
              ease: "linear"
            }}
          >
            <Cloud className="text-white" size={48 + i * 24} />
          </motion.div>
        ))}

        {/* Green Lands */}
        <div className="absolute bottom-0 left-0 w-full">
          <svg viewBox="0 0 1440 320" className="w-full">
            <path fill="#4CAF50" fillOpacity="1" d="M0,224L48,213.3C96,203,192,181,288,181.3C384,181,480,203,576,213.3C672,224,768,224,864,213.3C960,203,1056,181,1152,181.3C1248,181,1344,203,1392,213.3L1440,224L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
          </svg>
        </div>

        {/* Trees */}
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute bottom-0"
            style={{ left: `${i * 25}%`, transform: `scale(${0.5 + Math.random() * 0.5})` }}
            animate={{ y: [0, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity, repeatType: "reverse" }}
          >
            <Tree className="text-green-700" size={64} />
          </motion.div>
        ))}
      </div>

      <div className="container mx-auto py-6 relative">
        <motion.div 
          className="text-center mb-8"
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-5xl font-bold text-purple-700 mb-2 font-comic">BrightPath Chat</h1>
          <p className="text-2xl text-blue-600 font-comic">Let's Talk and Learn Together!</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-6xl mx-auto">
          <motion.div
            initial={{ x: -100, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="bg-gradient-to-br from-yellow-300 to-orange-300 rounded-3xl shadow-xl overflow-hidden border-4 border-yellow-400"
          >
            <VideoSection 
              isRecording={isRecording}
              onStart={startRecording}
              onStop={stopRecording}
            />
          </motion.div>
          <motion.div
            initial={{ x: 100, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="bg-gradient-to-br from-pink-200 to-purple-300 rounded-3xl shadow-xl overflow-hidden border-4 border-pink-300"
          >
            <ChatSection 
              messages={messages}
              currentSpeech={currentSpeech}
              isProcessing={isProcessing}
              isRecording={isRecording}
            />
          </motion.div>
        </div>
      </div>
    </div>
  )
}