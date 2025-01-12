import React, { useState } from 'react'
import { ArrowRight } from 'lucide-react'

interface StoryInputProps {
  onSubmit: (storyData: string) => void; // Adjust the type as needed
}

export default function StoryInput({ onSubmit }: StoryInputProps) {
  const [input, setInput] = useState('')

  const handleSubmit = async () => {
    try {
      const response = await fetch('http://localhost:5000/StoryTeller', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: input }),
      })
      const data = await response.json()
      console.log('Response from server:', data)
      onSubmit(data) // Pass the server response to the parent
    } catch (error) {
      console.error('Error:', error)
    }
  }

  return (
    <div className="flex items-center mt-4 w-full max-w-2xl">
      <textarea
        className="flex-grow p-4 border-2 border-blue-400 rounded-l-lg focus:outline-none focus:border-blue-600 font-serif text-lg"
        placeholder="Once upon a time..."
        rows={3}
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />
      <button
        onClick={handleSubmit}
        className="bg-blue-500 text-white p-4 rounded-r-lg hover:bg-blue-600 focus:outline-none transition duration-300 ease-in-out transform hover:scale-105"
      >
        <ArrowRight size={24} />
      </button>
    </div>
  )
}