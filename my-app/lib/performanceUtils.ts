export interface PerformanceEntry {
    date: string;        // Date of the performance
    score: number;       // Final score
    time: number;        // Time taken to complete the game in seconds
    correctAnswers: number;  // Number of correct answers
    mistakes: number;        // Number of mistakes made
    highestScore: number;    // Highest score achieved
    attempts: number;        // Number of attempts made
}

// Function to save performance data with date for multiple games
export function savePerformance(
    gameName: string, 
    score: number, 
    time: number, 
    correctAnswers: number, 
    mistakes: number, 
    highestScore: number, 
    attempts: number
) {
    try {
        const currentDate = new Date().toISOString().split('T')[0]; // Get current date in YYYY-MM-DD format
        const performanceKey = `${gameName}_performance`;

        // Get existing performance data from localStorage
        const existingData: PerformanceEntry[] = JSON.parse(localStorage.getItem(performanceKey) || '[]');

        // Check if data already exists for the current date and update it
        const existingEntryIndex = existingData.findIndex(entry => entry.date === currentDate);
        if (existingEntryIndex !== -1) {
            existingData[existingEntryIndex] = {
                date: currentDate,
                score,
                time,
                correctAnswers,
                mistakes,
                highestScore: Math.max(existingData[existingEntryIndex].highestScore, highestScore), // Keep the highest score
                attempts: existingData[existingEntryIndex].attempts + attempts, // Increment attempts
            };
        } else {
            // Add new performance entry for the current day
            existingData.push({
                date: currentDate,
                score,
                time,
                correctAnswers,
                mistakes,
                highestScore,
                attempts,
            });
        }

        // Save the updated performance data back to localStorage
        localStorage.setItem(performanceKey, JSON.stringify(existingData));
    } catch (error) {
        console.error("Error saving performance data:", error);
    }
}

// Function to retrieve performance data for a specific game
export function getPerformance(gameName: string): PerformanceEntry[] {
    try {
        const performanceKey = `${gameName}_performance`;
        const performanceData: PerformanceEntry[] = JSON.parse(localStorage.getItem(performanceKey) || '[]');
        return performanceData;
    } catch (error) {
        console.error("Error retrieving performance data:", error);
        return [];
    }
}

// Function to calculate average time for a specific game
export function getAverageTime(gameName: string): number {
    const performanceData = getPerformance(gameName);
    if (performanceData.length === 0) return 0;

    const totalTime = performanceData.reduce((sum, entry) => sum + entry.time, 0);
    return totalTime / performanceData.length;
}

// Function to calculate average number of correct answers for a specific game
export function getAverageCorrectAnswers(gameName: string): number {
    const performanceData = getPerformance(gameName);
    if (performanceData.length === 0) return 0;

    const totalCorrectAnswers = performanceData.reduce((sum, entry) => sum + entry.correctAnswers, 0);
    return totalCorrectAnswers / performanceData.length;
}

// Function to calculate average number of mistakes for a specific game
export function getAverageMistakes(gameName: string): number {
    const performanceData = getPerformance(gameName);
    if (performanceData.length === 0) return 0;

    const totalMistakes = performanceData.reduce((sum, entry) => sum + entry.mistakes, 0);
    return totalMistakes / performanceData.length;
}

// Function to calculate highest score for a specific game
export function getHighestScore(gameName: string): number {
    const performanceData = getPerformance(gameName);
    if (performanceData.length === 0) return 0;

    return performanceData.reduce((max, entry) => Math.max(max, entry.highestScore), 0);
}

// Function to calculate total attempts for a specific game
export function getTotalAttempts(gameName: string): number {
    const performanceData = getPerformance(gameName);
    if (performanceData.length === 0) return 0;

    return performanceData.reduce((sum, entry) => sum + entry.attempts, 0);
}
