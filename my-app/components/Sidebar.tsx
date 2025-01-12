import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Link from 'next/link'
import { ChevronRight, ChevronLeft, Home, Gamepad2, Book, BarChart2, Settings } from 'lucide-react'

const navItems = [
  // { name: 'SignUp', icon: Home, href: '/UserAuth/signup' },
  // { name: 'Login', icon: Home, href: '/UserAuth/login' },
  { name: 'Home', icon: Home, href: '/' },
  { name: 'Games', icon: Gamepad2, href: '/Games' },
  { name: 'Stories', icon: Book, href: '/story-creator' },
  { name: 'Learn', icon: Book, href: '/Learn_chatbot' },
  { name: 'Communicate', icon: Book, href: '/Communication' },
  { name: 'Dashboard', icon: BarChart2, href: '/Dashboard' },
  
]

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <motion.div
      className="fixed top-0 left-0 bottom-0 z-50 flex"
      initial={{ width: '40px' }}
      animate={{ width: isOpen ? '200px' : '40px' }}
    >
      <motion.div
        className="h-full bg-blue-500 text-white overflow-hidden"
        animate={{ width: isOpen ? '200px' : '0px' }}
      >
        <nav className="mt-16 w-48">
          <AnimatePresence>
            {isOpen && navItems.map((item, index) => (
              <motion.div
                key={item.name}
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ delay: index * 0.1 }}
              >
                <Link href={item.href} passHref>
                  <motion.div className="flex items-center py-2 px-4 hover:bg-blue-600 whitespace-nowrap">
                    <item.icon className="mr-2" size={20} />
                    {item.name}
                  </motion.div>
                </Link>
              </motion.div>
            ))}
          </AnimatePresence>
        </nav>
      </motion.div>
      <button
        className="absolute top-4 right-0 w-10 h-10 bg-blue-500 text-white rounded-r-full flex items-center justify-center"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <ChevronLeft size={24} /> : <ChevronRight size={24} />}
      </button>
    </motion.div>
  )
}
