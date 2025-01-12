import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const data = [
  { day: 'Mon', sessions: 3 },
  { day: 'Tue', sessions: 2 },
  { day: 'Wed', sessions: 3 },
  { day: 'Thu', sessions: 1 },
  { day: 'Fri', sessions: 2 },
  { day: 'Sat', sessions: 4 },
  { day: 'Sun', sessions: 3 },
];

const ConsistencyCard = () => {
  return (
    <motion.div
      className="bg-white p-6 rounded-2xl shadow-lg border-4 border-[#FFD700]"
      whileHover={{ scale: 1.05, rotate: -1 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <h2 className="text-2xl font-bold mb-4 text-[#FFD700] text-center">Consistency</h2>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="sessions" fill="#FFD700" />
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-4 bg-[#FFFACD] p-4 rounded-xl">
        <p className="font-semibold text-[#DAA520]">Weekly Sessions: 18</p>
        <p className="font-semibold text-[#DAA520]">Completion Rate: 85%</p>
      </div>
    </motion.div>
  );
};

export default ConsistencyCard;

