import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'BrightPath Games',
  description: 'Fun learning games for children with autism',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-b from-sky-300 via-sky-200 to-green-200 relative overflow-hidden">
          {/* Decorative elements */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute h-32 w-32 rounded-full bg-green-500/20 -left-16 bottom-20" />
            <div className="absolute h-24 w-24 rounded-full bg-green-500/20 left-32 bottom-28" />
            <div className="absolute h-40 w-40 rounded-full bg-green-500/20 right-20 bottom-16" />
            {/* Stars */}
            {Array.from({ length: 20 }).map((_, i) => (
              <div
                key={i}
                className="absolute w-1 h-1 bg-yellow-300 rounded-full animate-twinkle"
                style={{
                  top: `${Math.random() * 100}%`,
                  left: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 2}s`,
                }}
              />
            ))}
          </div>
          {children}
        </div>
      </body>
    </html>
  )
}

