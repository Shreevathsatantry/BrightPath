import { motion } from 'framer-motion';
import { Star } from 'lucide-react';

const milestones = [
  { name: 'First Quiz Completed', achieved: true },
  { name: '10 Games Played', achieved: true },
  { name: 'Vocabulary Milestone', achieved: false },
  { name: '1 Month Streak', achieved: true },
  { name: 'Problem-Solving Master', achieved: false },
];

const MilestonesCard = () => {
  return (
    <motion.div
      className="bg-white p-6 rounded-2xl shadow-lg border-4 border-[#FFD700]"
      whileHover={{ scale: 1.05, rotate: 1 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <h2 className="text-2xl font-bold mb-4 text-[#FFD700] text-center">Milestones</h2>
      <ul className="space-y-2">
        {milestones.map((milestone, index) => (
          <motion.li
            key={index}
            className="flex items-center space-x-2"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Star
              className={`w-6 h-6 ${
                milestone.achieved ? 'text-[#FFD700]' : 'text-gray-300'
              }`}
            />
            <span
              className={`${
                milestone.achieved ? 'text-black' : 'text-gray-500'
              }`}
            >
              {milestone.name}
            </span>
          </motion.li>
        ))}
      </ul>
      <div className="mt-4 bg-[#FFFACD] p-4 rounded-xl">
        <p className="font-semibold text-[#DAA520]">Milestones Achieved: 3/5</p>
        <p className="font-semibold text-[#DAA520]">Next Milestone: Vocabulary Milestone</p>
      </div>
    </motion.div>
  );
};

export default MilestonesCard;

