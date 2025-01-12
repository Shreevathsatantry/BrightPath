import { motion } from 'framer-motion';
import { Smile, Meh, Frown, Star } from 'lucide-react';

interface Emotion {
  type: 'happy' | 'neutral' | 'sad';
  count: number;
  percentage: number;
}

interface EmotionTrackerProps {
  emotions: Emotion[];
  trend: 'positive' | 'neutral' | 'negative';
  focusTime: number;
  responseTime: number;
}

const emotionIcons = {
  happy: Smile,
  neutral: Meh,
  sad: Frown
};

const emotionColors = {
  happy: '#4ADE80',
  neutral: '#FCD34D',
  sad: '#FB7185'
};

export function EmotionTracker({ emotions, trend, focusTime, responseTime }: EmotionTrackerProps) {
  return (
    <motion.div
      className="bg-white rounded-xl p-6 shadow-sm"
      whileHover={{ scale: 1.01 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <h3 className="font-medium text-gray-900 mb-4">Emotional Development</h3>
      <div className="grid grid-cols-3 gap-4 mb-6">
        {emotions.map((emotion, index) => {
          const Icon = emotionIcons[emotion.type];
          return (
            <motion.div
              key={emotion.type}
              className="text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <motion.div
                className="w-12 h-12 rounded-full mx-auto flex items-center justify-center"
                style={{ backgroundColor: emotionColors[emotion.type] }}
                whileHover={{ scale: 1.1 }}
                transition={{ duration: 0.5 }}
              >
                <Icon className="w-6 h-6 text-white" />
              </motion.div>
              <p className="text-sm font-medium mt-2">{emotion.count}</p>
              <p className="text-xs text-gray-500">{emotion.percentage}%</p>
            </motion.div>
          );
        })}
      </div>
      <div className="space-y-4">
        <motion.div 
          className="flex justify-between items-center"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <span className="text-sm text-gray-500">Focus Time</span>
          <span className="text-sm font-medium">{focusTime} min</span>
        </motion.div>
        <motion.div 
          className="flex justify-between items-center"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <span className="text-sm text-gray-500">Response Time</span>
          <span className="text-sm font-medium">{responseTime} sec</span>
        </motion.div>
        <motion.div 
          className="flex justify-between items-center"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <span className="text-sm text-gray-500">Trend</span>
          <div className="flex items-center gap-1">
            <Star className={`w-4 h-4 ${
              trend === 'positive' ? 'text-yellow-400' : 'text-gray-300'
            }`} />
            <Star className={`w-4 h-4 ${
              trend === 'positive' ? 'text-yellow-400' : 'text-gray-300'
            }`} />
            <Star className={`w-4 h-4 ${
              trend === 'positive' ? 'text-yellow-400' : 'text-gray-300'
            }`} />
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}

