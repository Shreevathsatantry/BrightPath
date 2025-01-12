import { motion } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

const data = [
  { name: 'Games', value: 40 },
  { name: 'Learning', value: 35 },
  { name: 'Communication', value: 25 },
];

const COLORS = ['#FF69B4', '#7CFC00', '#87CEEB'];

const TimeSpentCard = () => {
  return (
    <motion.div
      className="bg-white p-6 rounded-2xl shadow-lg border-4 border-[#FF69B4]"
      whileHover={{ scale: 1.05, rotate: 1 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <h2 className="text-2xl font-bold mb-4 text-[#FF69B4] text-center">Time Spent</h2>
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
            label
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="mt-4 bg-[#FFF0F5] p-4 rounded-xl">
        <p className="font-semibold text-[#FF1493]">Total Time: 10 hours/week</p>
        <p className="font-semibold text-[#FF1493]">Focus Time: 8 hours/week</p>
      </div>
    </motion.div></motion.div>
  );
};

export default TimeSpentCard;

