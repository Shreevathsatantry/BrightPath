import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const data = [
  { month: 'Jan', understanding: 30 },
  { month: 'Feb', understanding: 45 },
  { month: 'Mar', understanding: 60 },
  { month: 'Apr', understanding: 80 },
];

const UnderstandingCard = () => {
  return (
    <motion.div
      className="bg-white p-6 rounded-2xl shadow-lg border-4 border-[#7CFC00]"
      whileHover={{ scale: 1.05, rotate: -1 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <h2 className="text-2xl font-bold mb-4 text-[#7CFC00] text-center">Understanding</h2>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="month" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="understanding" stroke="#7CFC00" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-4 bg-[#F0FFF0] p-4 rounded-xl">
        <p className="font-semibold text-[#32CD32]">Topics Learned: 15</p>
        <p className="font-semibold text-[#32CD32]">Comprehension Rate: 85%</p>
      </div>
    </motion.div>
  );
};

export default UnderstandingCard;

