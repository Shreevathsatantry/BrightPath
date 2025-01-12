import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const data = [
  { name: 'Game 1', success: 80 },
  { name: 'Game 2', success: 65 },
  { name: 'Game 3', success: 90 },
  { name: 'Quiz', success: 75 },
];

const AnalyticalSkillsCard = () => {
  return (
    <motion.div
      className="bg-white p-6 rounded-2xl shadow-lg border-4 border-[#FF69B4]"
      whileHover={{ scale: 1.05, rotate: 1 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <h2 className="text-2xl font-bold mb-4 text-[#FF69B4] text-center">Analytical Skills</h2>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="success" fill="#FF69B4" />
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-4 bg-[#FFF0F5] p-4 rounded-xl">
        <p className="font-semibold text-[#FF69B4]">Average Performance: 77.5%</p>
        <p className="font-semibold text-[#FF69B4]">Cognitive Growth: +15%</p>
      </div>
    </motion.div>
  );
};

export default AnalyticalSkillsCard;

