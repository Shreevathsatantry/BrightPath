"use client";

import { motion } from 'framer-motion';
import { Brain, BookOpen, MessageSquare, Clock, Target, Lightbulb, Activity, Mic, PenTool } from 'lucide-react';
import { MetricCard } from './MetricCard';
import { ProgressChart } from './ProgressChart';
import { MilestoneCard } from './MilestoneCard';
import { EmotionTracker } from './EmotionTracker';
import { CommunicationCard } from './CommunicationCard';
import { AISuggestionsCard } from './AISuggestionsCard';
import Background from './Background_copy';
import Sidebar from './Sidebar';
import axios from 'axios';
import { useState, useEffect } from 'react';

export default function Dashboard() {
  const [aiSuggestions, setAiSuggestions] = useState([]);
  const [allGameTotalAverageScore, setAllGameTotalAverageScore] = useState(0);
  const [dayStreak, setDayStreak] = useState(0);
  const [analyticalData, setAnalyticalData] = useState([
    { name: 'Game 1', value: 0, color: '#4ADE80' },
    { name: 'Game 2', value: 0, color: '#FCD34D' },
    { name: 'Game 3', value: 0, color: '#4ADE80' },
    { name: 'Quiz', value: 0, color: '#FCD34D' },
  ]);
  const [consistencyData, setConsistencyData] = useState([
    { day: "Mon", value: 0 },
    { day: "Tue", value: 0 },
    { day: "Wed", value: 0 },
    { day: "Thu", value: 0 },
    { day: "Fri", value: 0 },
    { day: "Sat", value: 0 },
    { day: "Sun", value: 0 },
  ]);
  useEffect(() => {
    const calculateDayStreak = () => {
      const today = new Date().toISOString().split('T')[0]; // Current date in YYYY-MM-DD format
      let streak = 0;

      const isDateInStorage = (date) => {
        return Object.keys(localStorage).some((key) => {
          const storedData = JSON.parse(localStorage.getItem(key) || '[]');
          return Array.isArray(storedData) && storedData.some((entry) => entry.date === date);
        });
      };

      if (isDateInStorage(today)) {
        streak = 1;
        let previousDate = new Date(today);

        while (true) {
          previousDate.setDate(previousDate.getDate() - 1); // Go to the previous day
          const prevDateStr = previousDate.toISOString().split('T')[0];

          if (isDateInStorage(prevDateStr)) {
            streak += 1;
          } else {
            break;
          }
        }
      }

      setDayStreak(streak);
    };

    calculateDayStreak();
  }, []);
  useEffect(() => {
    const getCurrentWeekDates = () => {
      const today = new Date();
      const dayOfWeek = today.getDay(); // Sunday - Saturday: 0 - 6
      const startOfWeek = new Date(today);
      startOfWeek.setDate(today.getDate() - dayOfWeek + 1); // Set to Monday
  
      const dates = [];
      for (let i = 0; i < 7; i++) {
        const date = new Date(startOfWeek);
        date.setDate(startOfWeek.getDate() + i);
        dates.push({
          day: date.toLocaleDateString("en-US", { weekday: "short" }), // Convert to day name
          date: date.toISOString().split("T")[0], // Convert to YYYY-MM-DD
        });
      }
  
      return dates;
    };
  
    const weekDates = getCurrentWeekDates();
    const updatedConsistencyData = consistencyData.map((dayEntry) => {
      const matchingDate = weekDates.find((entry) => entry.day === dayEntry.day);
      if (!matchingDate) return dayEntry;
  
      let totalAverageScore = 0;
  
      Object.keys(localStorage).forEach((key) => {
        const storedData = JSON.parse(localStorage.getItem(key) || "[]");
        if (Array.isArray(storedData)) {
          storedData.forEach((entry) => {
            if (entry.date === matchingDate.date) {
              totalAverageScore += parseFloat((entry.TotalAverageScore) || 0);
            }
          });
        }
      });
  
      return { ...dayEntry, value: totalAverageScore };
    });
  
    setConsistencyData(updatedConsistencyData);
  }, [consistencyData]);
  
  
  useEffect(() => {
    // Fetch AI suggestions
    const fetchAiSuggestions = async () => {
      try {
        const response = await axios.get("http://localhost:5000/AiSuggestionBot");
        
        const rawSuggestions = response.data.response;
  
        const suggestions = rawSuggestions
          .split('\n\n') // Split suggestions by double newline
          .map((suggestion, index) => {
            const [categoryText, ...suggestionTextParts] = suggestion.split(':');
            const suggestionText = suggestionTextParts.join(':').trim(); // Join if there were any extra colons in the suggestion part
  
            return { 
              id: String(index + 1), 
              suggestion: suggestionText.replace(/\*\*/g, '').trim(),  // Remove ** from suggestion
              category: categoryText.replace(/\*/g, '').trim() // Remove * from category
            };
          });
        
        console.log("AI suggestions: ", suggestions);
        
        setAiSuggestions(suggestions);
      } catch (error) {
        console.error("Error fetching AI suggestions:", error);
      }
    };
  
    fetchAiSuggestions();
  
    // Keys representing different game categories
    const gameScoreKeys = [
      { key: 'musicalGameScore', name: 'Musical Game' },
      { key: 'emotionGameScores', name: 'Emotion Game' },
      { key: 'colorMatchingGameScores', name: 'Color Game' },
      { key: 'ShapeSortingGame', name: 'Shape Sort' },
      {key : 'grammarDetectiveGame' , name:'Grammar'},
      {key : 'basicArithmeticGame' , name:'Math'}
    ];
  
    let totalAverageScore = 0;
    let totalEntries = 0;
    const updatedAnalyticalData = [];
  
    // Loop through each game category to compute individual averages
    gameScoreKeys.forEach(({ key, name }, index) => {
      const storedData = JSON.parse(localStorage.getItem(key) || '[]'); // Fetch the stored data for each key
  
      let categoryTotal = 0;
      let categoryCount = 0;
  
      if (storedData && Array.isArray(storedData)) {
        storedData.forEach((entry) => {
          categoryTotal += parseFloat(entry.TotalAverageScore);
          categoryCount += 1;
        });
      }
  
      const categoryAverage = categoryCount > 0 ? (categoryTotal / categoryCount).toFixed(2) : 0;
  
      updatedAnalyticalData.push({
        name,
        value: parseFloat(categoryAverage) * 100,
        color: index % 2 === 0 ? '#4ADE80' : '#FCD34D',
      });
  
      totalAverageScore += categoryTotal;
      totalEntries += categoryCount;
    });
  
    // Update the analytical data state
    setAnalyticalData(updatedAnalyticalData);
  
    // Calculate the overall average score
    const calculatedAverage = totalEntries > 0 ? (totalAverageScore / totalEntries).toFixed(4) : 0;
    setAllGameTotalAverageScore(calculatedAverage * 100);
  }, []);

  const [understandingData, setUnderstandingData] = useState([
    { month: 'Jan', value: 30 },
    { month: 'Feb', value: 45 },
    { month: 'Mar', value: 60 },
    { month: 'Apr', value: 10 },
  ]);

  useEffect(() => {
    const getRandomNumber = () => {
      return Math.floor(Math.random() * (80 - 30 + 1)) + 30;
    };

    // Update random values for understandingData every time the page loads
    const updatedUnderstandingData = understandingData.map(data => ({
      ...data,
      value: getRandomNumber(),
    }));
    setUnderstandingData(updatedUnderstandingData);
  }, []);
  
  const daysStreak = [
    { name: 'Games', value: 40, color: '#FF4B91' },
    { name: 'Learning', value: 35, color: '#65B741' },
    { name: 'Communication', value: 25, color: '#4477CE' },
  ];
  

  const milestones = [
    {
      id: '1',
      title: 'First Quiz Completed',
      achieved: true,
      icon: 'trophy' as const,
      date: '2 days ago'
    },
    {
      id: '2',
      title: '10 Games Played',
      achieved: true,
      icon: 'star' as const,
      date: '1 week ago'
    },
    {
      id: '3',
      title: 'Vocabulary Master',
      achieved: false,
      icon: 'award' as const
    },
  ];
  
  const emotions = [
    { type: 'happy' as const, count: 15, percentage: 75 },
    { type: 'neutral' as const, count: 4, percentage: 20 },
    { type: 'sad' as const, count: 1, percentage: 5 },
  ];
  
  const [communicationSkills, setCommunicationSkills] = useState([
    { name: 'Verbal Skills', value: 85, icon: Mic },
    { name: 'Written Skills', value: 78, icon: PenTool },
    { name: 'Comprehension', value: 90, icon: BookOpen },
  ]);

  useEffect(() => {
    const getRandomNumber = () => {
      return Math.floor(Math.random() * (85 - 50 + 1)) + 50;
    };

    // Update random values every time the page loads
    const updatedCommunicationSkills = communicationSkills.map(skill => ({
      ...skill,
      value: getRandomNumber(),
    }));
    setCommunicationSkills(updatedCommunicationSkills);
  }, []);
  


  return (
    <div className="min-h-screen relative">
      <Background />
      <Sidebar />
      <div className="relative z-10 p-8">
        <div className="max-w-7xl mx-auto space-y-6">
          {/* Title */}
          <motion.h1 
            className="text-4xl font-bold text-gray-800 text-center mb-8"
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            Child Development Dashboard
          </motion.h1>

          {/* Top Metrics */}
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <MetricCard
              title="Analytical Skills"
              value={allGameTotalAverageScore}
              icon={Brain}
              color="#FF4B91"
              subtitle="AverageScore"
              unit="%"
            />
            <MetricCard
              title="Understanding"
              value={78}
              icon={Lightbulb}
              color="#65B741"
              subtitle="15 Topics Learned"
              unit="%"
            />
            <MetricCard
              title="Communication"
              value={92}
              icon={MessageSquare}
              color="#4477CE"
              subtitle="2nd Grade Level"
              unit="%"
            />
            <MetricCard
              title="Streak"
              value={dayStreak}
              icon={Clock}
              color="#FFB534"
              subtitle={`${dayStreak} days of streak`} // Correct template literal syntax
            />

          </motion.div>

          {/* Charts Row 1 */}
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <ProgressChart
              title="Analytical Performance"
              data={analyticalData}
              color="#FF4B91"
              type="bar"
              dataKey="value"
              xAxisKey="name"
              height={200}
              lastUpdated="updated now"
              tooltipFormatter={(value) => `${value}%`}
            />
            <ProgressChart
              title="Understanding Progress"
              data={understandingData}
              color="#65B741"
              type="line"
              dataKey="value"
              xAxisKey="month"
              height={200}
              lastUpdated="today"
              tooltipFormatter={(value) => `${value}%`}
            />
          </motion.div>

          {/* Communication and Weekly Consistency */}
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <CommunicationCard skills={communicationSkills} />
            <ProgressChart
              title="Weekly Consistency"
              data={consistencyData}
              color="#FFB534"
              type="bar"
              dataKey="value"
              xAxisKey="day"
              height={200}
              lastUpdated="just now"
              tooltipFormatter={(value) => `${value} sessions`}
            />
          </motion.div>

          {/* Charts Row 2 */}
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.8 }}
          >
            <ProgressChart
              title="Time Distribution"
              data={daysStreak}
              color="#4477CE"
              type="pie"
              dataKey="value"
              height={200}
              lastUpdated="updated now"
              tooltipFormatter={(value) => `${value}%`}
            />
            <AISuggestionsCard suggestions={aiSuggestions} />
          </motion.div>

          {/* Bottom Row */}
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 1.0 }}
          >
            <EmotionTracker
              emotions={emotions}
              trend="positive"
              focusTime={45}
              responseTime={2.5}
            />
            <MilestoneCard milestones={milestones} />
          </motion.div>
        </div>
      </div>
    </div>
  );
}