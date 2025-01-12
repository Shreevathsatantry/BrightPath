import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const data = [
  { emotion: 'Happy', frequency: 70 },
  { emotion: 'Excited', frequency: 50 },
  { emotion: 'Calm', frequency: 60 },
  { emotion: 'Frustrated', frequency: 20 },
  { emotion: 'Confused', frequency: 30 },
];

const EmotionalDevelopmentCard = () => {
  return (
    <motion.div
      className="bg-white p-6 rounded-2xl shadow-lg border-4 border-[#87CEEB]"
      whileHover={{ scale: 1.05, rotate: -1 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <h2 className="text-2xl font-bold mb-4 text-[#87CEEB] text-center">Emotional Development</h2>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="emotion" type="category" />
          <Tooltip />
          <Bar dataKey="frequency" fill="#87CEEB" />
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-4 bg-[#F0F8FF] p-4 rounded-xl">
        <p className="font-semibold text-[#4682B4]">Positive Emotions: 75%</p>
        <p className="font-semibold text-[#4682B4]">Emotional Growth: +20%</p>
      </div>
    </motion.div>
  );
};

export default EmotionalDevelopmentCard;

