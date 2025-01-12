import { motion } from 'framer-motion';
import { Check } from 'lucide-react';

interface Activity {
  id: string;
  title: string;
  time: string;
  completed: boolean;
}

interface ActivityCardProps {
  activities: Activity[];
}

export function ActivityCard({ activities }: ActivityCardProps) {
  return (
    <motion.div
      className="bg-white rounded-xl p-6 shadow-sm"
      whileHover={{ scale: 1.01 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <h3 className="font-medium text-gray-900 mb-4">Recent Activities</h3>
      <div className="space-y-4">
        {activities.map((activity) => (
          <div key={activity.id} className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 rounded-full bg-green-100 flex items-center justify-center">
                <Check className="w-3 h-3 text-green-600" />
              </div>
              <span className="text-sm text-gray-600">{activity.title}</span>
            </div>
            <span className="text-sm text-gray-400">{activity.time}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

