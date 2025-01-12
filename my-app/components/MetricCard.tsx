import { motion } from 'framer-motion';
import { TypeIcon as type, type LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: number;
  icon: LucideIcon;
  color: string;
  subtitle?: string;
  unit?: string;
}

export function MetricCard({ title, value, icon: Icon, color, subtitle, unit = '' }: MetricCardProps) {
  return (
    <motion.div
      className="bg-white rounded-xl p-4 shadow-sm hover:shadow-lg transition-shadow"
      whileHover={{ scale: 1.05 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm text-gray-500 font-medium">{title}</h3>
          <motion.p 
            className="text-2xl font-bold mt-1"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {value}{unit}
          </motion.p>
          {subtitle && (
            <p className="text-sm text-gray-400 mt-1">{subtitle}</p>
          )}
        </div>
        <motion.div
          className={`w-10 h-10 rounded-lg flex items-center justify-center`}
          style={{ backgroundColor: color }}
          whileHover={{ rotate: 360 }}
          transition={{ duration: 0.5 }}
        >
          <Icon className="w-5 h-5 text-white" />
        </motion.div>
      </div>
    </motion.div>
  );
}

