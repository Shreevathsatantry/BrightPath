import { motion } from 'framer-motion';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from 'recharts';

interface ProgressCardProps {
  title: string;
  data: any[];
  color: string;
  type: 'line' | 'bar';
  dataKey: string;
  xAxisKey: string;
  lastUpdated: string;
}

export function ProgressCard({ title, data, color, type, dataKey, xAxisKey, lastUpdated }: ProgressCardProps) {
  return (
    <motion.div
      className="rounded-xl shadow-sm overflow-hidden"
      style={{ backgroundColor: color }}
      whileHover={{ scale: 1.02 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <div className="p-4 h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          {type === 'line' ? (
            <LineChart data={data}>
              <XAxis 
                dataKey={xAxisKey} 
                stroke="rgba(255,255,255,0.5)"
                tick={{ fill: 'white', fontSize: 12 }}
              />
              <YAxis 
                stroke="rgba(255,255,255,0.5)"
                tick={{ fill: 'white', fontSize: 12 }}
              />
              <Line
                type="monotone"
                dataKey={dataKey}
                stroke="white"
                strokeWidth={2}
                dot={{ fill: 'white' }}
              />
            </LineChart>
          ) : (
            <BarChart data={data}>
              <XAxis 
                dataKey={xAxisKey}
                stroke="rgba(255,255,255,0.5)"
                tick={{ fill: 'white', fontSize: 12 }}
              />
              <YAxis 
                stroke="rgba(255,255,255,0.5)"
                tick={{ fill: 'white', fontSize: 12 }}
              />
              <Bar dataKey={dataKey} fill="white" />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
      <div className="bg-white p-4">
        <h3 className="font-medium text-gray-900">{title}</h3>
        <p className="text-sm text-gray-500 mt-1">
          Updated {lastUpdated}
        </p>
      </div>
    </motion.div>
  );
}

