import { motion } from 'framer-motion';
import { MessageCircle, Mic, PenTool, BookOpen } from 'lucide-react';

interface CommunicationSkill {
  name: string;
  value: number;
  icon: React.ElementType;
}

interface CommunicationCardProps {
  skills: CommunicationSkill[];
}

export function CommunicationCard({ skills }: CommunicationCardProps) {
  return (
    <motion.div
      className="bg-white rounded-xl p-6 shadow-sm border-t-4 border-[#4477CE]"
      whileHover={{ scale: 1.01 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <h3 className="font-medium text-gray-900 mb-4 flex items-center">
        <MessageCircle className="w-5 h-5 mr-2 text-[#4477CE]" />
        Communication Skills Analysis
      </h3>
      <div className="space-y-4">
        {skills.map((skill) => (
          <motion.div
            key={skill.name}
            className="flex items-center gap-3 p-3 rounded-lg bg-[#E5F0FF]"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <skill.icon className="w-5 h-5 text-[#4477CE]" />
            <div className="flex-grow">
              <p className="text-sm font-medium text-[#2C5282]">{skill.name}</p>
              <div className="w-full bg-[#B8D3FF] rounded-full h-2.5 mt-2">
                <div 
                  className="bg-[#4477CE] h-2.5 rounded-full" 
                  style={{ width: `${skill.value}%` }}
                ></div>
              </div>
            </div>
            <span className="text-sm font-medium text-[#4477CE]">{skill.value}%</span>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

