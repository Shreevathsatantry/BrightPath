import { motion } from 'framer-motion';
import { Star, Trophy, Award } from 'lucide-react';

interface Milestone {
  id: string;
  title: string;
  achieved: boolean;
  icon: 'star' | 'trophy' | 'award';
  date?: string;
}

interface MilestoneCardProps {
  milestones: Milestone[];
}

const icons = {
  star: Star,
  trophy: Trophy,
  award: Award
};

export function MilestoneCard({ milestones }: MilestoneCardProps) {
  return (
    <motion.div
      className="bg-white rounded-xl p-6 shadow-sm"
      whileHover={{ scale: 1.01 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <h3 className="font-medium text-gray-900 mb-4">Milestones & Achievements</h3>
      <div className="space-y-4">
        {milestones.map((milestone, index) => {
          const Icon = icons[milestone.icon];
          return (
            <motion.div
              key={milestone.id}
              className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-50"
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <motion.div 
                className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  milestone.achieved ? 'bg-yellow-100' : 'bg-gray-100'
                }`}
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
              >
                <Icon className={`w-4 h-4 ${
                  milestone.achieved ? 'text-yellow-600' : 'text-gray-400'
                }`} />
              </motion.div>
              <div>
                <p className={`font-medium ${
                  milestone.achieved ? 'text-gray-900' : 'text-gray-500'
                }`}>
                  {milestone.title}
                </p>
                {milestone.date && (
                  <p className="text-sm text-gray-400">{milestone.date}</p>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}

