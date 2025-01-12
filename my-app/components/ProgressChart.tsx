import { motion } from 'framer-motion';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, PieChart, Pie, Cell } from 'recharts';

interface ProgressChartProps {
  title: string;
  data: any[];
  color: string;
  type: 'line' | 'bar' | 'pie';
  dataKey: string;
  xAxisKey?: string;
  lastUpdated: string;
  height?: number;
  tooltipFormatter?: (value: number) => string;
}

export function ProgressChart({ 
  title, 
  data, 
  color, 
  type, 
  dataKey, 
  xAxisKey, 
  lastUpdated, 
  height = 200,
  tooltipFormatter 
}: ProgressChartProps) {
  const renderChart = () => {
    switch (type) {
      case 'line':
        return (
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
            <Tooltip 
              formatter={tooltipFormatter}
              contentStyle={{ backgroundColor: 'white', border: 'none' }}
            />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke="white"
              strokeWidth={2}
              dot={{ fill: 'white' }}
            />
          </LineChart>
        );
      case 'bar':
        return (
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
            <Tooltip 
              formatter={tooltipFormatter}
              contentStyle={{ backgroundColor: 'white', border: 'none' }}
            />
            <Bar dataKey={dataKey} fill="white" />
          </BarChart>
        );
      case 'pie':
        return (
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              fill="#8884d8"
              paddingAngle={5}
              dataKey={dataKey}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color || color} />
              ))}
            </Pie>
            <Tooltip 
              formatter={tooltipFormatter}
              contentStyle={{ backgroundColor: 'white', border: 'none' }}
            />
          </PieChart>
        );
    }
  };

  return (
    <motion.div
      className="rounded-xl shadow-sm overflow-hidden flex flex-col"
      style={{ backgroundColor: color }}
      whileHover={{ scale: 1.02 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <motion.div 
        className="p-4 flex-grow" 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h3 className="font-medium text-white mb-2">{title}</h3>
        <div style={{ height: height }}>
          <ResponsiveContainer width="100%" height="100%">
            {renderChart()}
          </ResponsiveContainer>
        </div>
      </motion.div>
      <div className="bg-white p-4">
        <p className="text-sm text-gray-500">
          Updated {lastUpdated}
        </p>
      </div>
    </motion.div>
  );
}

