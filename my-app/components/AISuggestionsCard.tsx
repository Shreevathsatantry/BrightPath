import { motion } from 'framer-motion';
import { Lightbulb, ChevronRight } from 'lucide-react';

interface AISuggestion {
  id: string;
  suggestion: string;
  category: string;
}

interface AISuggestionsCardProps {
  suggestions: AISuggestion[];
}

export function AISuggestionsCard({ suggestions }: AISuggestionsCardProps) {
  return (
    <motion.div
      className="bg-white rounded-xl p-6 shadow-sm border-t-4 border-[#65B741]"
      whileHover={{ scale: 1.01 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <h3 className="font-medium text-gray-900 mb-4 flex items-center">
        <Lightbulb className="w-5 h-5 mr-2 text-[#65B741]" />
        AI Suggestions
      </h3>
      <div className="space-y-4">
        {suggestions.map((suggestion) => (
          <motion.div
            key={suggestion.id}
            className="flex items-start gap-3 p-3 rounded-lg bg-[#E8F5E9]"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex-grow">
              <p className="text-sm text-[#2E7D32]">{suggestion.suggestion}</p>
              <p className="text-xs text-[#4A5568] mt-1">{suggestion.category}</p>
            </div>
            <ChevronRight className="w-4 h-4 text-[#65B741] flex-shrink-0" />
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

