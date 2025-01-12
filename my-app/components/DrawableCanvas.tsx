'use client'

import React, { useRef, useEffect, useState } from 'react'

export default function DrawableCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const lastPositionRef = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas) {
      const context = canvas.getContext('2d')
      if (context) {
        context.fillStyle = '#FFFFFF'
        context.fillRect(0, 0, canvas.width, canvas.height)
      }
    }
  }, [])

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true)
    const canvas = canvasRef.current
    if (canvas) {
      const rect = canvas.getBoundingClientRect()
      lastPositionRef.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      }
    }
  }

  const stopDrawing = () => {
    setIsDrawing(false)
    lastPositionRef.current = null
  }

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return

    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (context && canvas) {
      const rect = canvas.getBoundingClientRect()
      const currentPosition = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      }

      context.lineWidth = 2
      context.lineCap = 'round'
      context.strokeStyle = '#000000'

      context.beginPath()
      if (lastPositionRef.current) {
        context.moveTo(lastPositionRef.current.x, lastPositionRef.current.y)
      } else {
        context.moveTo(currentPosition.x, currentPosition.y)
      }
      context.lineTo(currentPosition.x, currentPosition.y)
      context.stroke()

      lastPositionRef.current = currentPosition
    }
  }

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={400}
      className="border-4 border-yellow-400 rounded-lg shadow-lg cursor-crosshair"
      onMouseDown={startDrawing}
      onMouseUp={stopDrawing}
      onMouseMove={draw}
      onMouseLeave={stopDrawing}
    />
  )
}

