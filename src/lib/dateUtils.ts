import { format, parseISO, addDays, startOfWeek, endOfWeek, isSameDay, isBefore, isAfter, addMinutes, parse } from 'date-fns';

/**
 * Format a time string (HH:mm) to 12-hour format with AM/PM
 */
export const formatTime = (timeString: string): string => {
  if (!timeString) return '--:--';
  
  // Handle both 'HH:mm' and 'HH:mm:ss' formats
  const [hours, minutes] = timeString.split(':');
  const hour = parseInt(hours, 10);
  const mins = minutes?.substring(0, 2) || '00';
  
  const period = hour >= 12 ? 'PM' : 'AM';
  const displayHour = hour % 12 || 12; // Convert 0 to 12 for 12 AM
  
  return `${displayHour}:${mins.padStart(2, '0')} ${period}`;
};

/**
 * Format a date string to YYYY-MM-DD format
 */
export const formatDateString = (date: Date | string): string => {
  try {
    // If input is already a string in YYYY-MM-DD format, return as is
    if (typeof date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(date)) {
      return date;
    }
    
    // Create a new date object to avoid mutating the original
    const d = new Date(date);
    
    // Check if the date is valid
    if (isNaN(d.getTime())) {
      console.error('Invalid date provided to formatDateString:', date);
      // Return today's date as fallback
      const today = new Date();
      return `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    }
    
    // Use UTC methods to avoid timezone issues
    const year = d.getUTCFullYear();
    const month = String(d.getUTCMonth() + 1).padStart(2, '0');
    const day = String(d.getUTCDate()).padStart(2, '0');
    
    return `${year}-${month}-${day}`;
  } catch (error) {
    console.error('Error in formatDateString:', error);
    // Return today's date as fallback
    const today = new Date();
    return `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
  }
};

/**
 * Get an array of dates for the week containing the given date
 */
export const getWeekDates = (date: Date): Date[] => {
  try {
    // Create a copy of the input date to avoid modifying it
    const inputDate = new Date(date);
    
    // Get the day of the week (0 = Sunday, 1 = Monday, etc.)
    const dayOfWeek = inputDate.getUTCDay();
    
    // Calculate the date of the previous Sunday
    const sunday = new Date(inputDate);
    sunday.setUTCDate(inputDate.getUTCDate() - dayOfWeek);
    
    // Create an array to hold all 7 days of the week
    const weekDates: Date[] = [];
    
    // Generate all 7 days of the week starting from Sunday
    for (let i = 0; i < 7; i++) {
      // Create a new date for each day of the week using UTC methods
      const day = new Date(Date.UTC(
        sunday.getUTCFullYear(),
        sunday.getUTCMonth(),
        sunday.getUTCDate() + i,
        0, 0, 0, 0  // Set time to midnight UTC
      ));
      
      weekDates.push(day);
    }
    
    return weekDates;
  } catch (error) {
    console.error('Error in getWeekDates:', error);
    // Return current week as fallback
    const today = new Date();
    const dayOfWeek = today.getUTCDay();
    const sunday = new Date(today);
    sunday.setUTCDate(today.getUTCDate() - dayOfWeek);
    
    const weekDates: Date[] = [];
    for (let i = 0; i < 7; i++) {
      const day = new Date(sunday);
      day.setUTCDate(sunday.getUTCDate() + i);
      weekDates.push(day);
    }
    
    return weekDates;
  }
};

/**
 * Check if two time slots overlap
 */
export const hasTimeOverlap = (
  start1: string, 
  end1: string, 
  start2: string, 
  end2: string,
  date: Date
): boolean => {
  try {
    // Parse the time strings into Date objects on the same date
    const parseTime = (timeStr: string) => {
      const [hours, minutes] = timeStr.split(':').map(Number);
      const dateObj = new Date(date);
      dateObj.setHours(hours, minutes || 0, 0, 0);
      return dateObj;
    };
    
    const startTime1 = parseTime(start1);
    const endTime1 = parseTime(end1);
    const startTime2 = parseTime(start2);
    const endTime2 = parseTime(end2);
    
    // Check if the time ranges overlap
    return startTime1 < endTime2 && endTime1 > startTime2;
  } catch (error) {
    console.error('Error in hasTimeOverlap:', error);
    return false;
  }
};

/**
 * Format a date range as a string
 */
export const formatDateRange = (startDate: Date | string, endDate: Date | string, formatStr = 'MMM d, yyyy'): string => {
  try {
    const start = typeof startDate === 'string' ? parseISO(startDate) : startDate;
    const end = typeof endDate === 'string' ? parseISO(endDate) : endDate;
    
    if (isSameDay(start, end)) {
      return format(start, formatStr);
    }
    
    return `${format(start, formatStr)} - ${format(end, formatStr)}`;
  } catch (error) {
    console.error('Error in formatDateRange:', error);
    return '';
  }
};

/**
 * Add minutes to a time string (HH:mm)
 */
export const addMinutesToTime = (timeStr: string, minutesToAdd: number): string => {
  try {
    const [hours, minutes] = timeStr.split(':').map(Number);
    const date = new Date();
    date.setHours(hours, minutes, 0, 0);
    
    const newDate = addMinutes(date, minutesToAdd);
    
    return `${String(newDate.getHours()).padStart(2, '0')}:${String(newDate.getMinutes()).padStart(2, '0')}`;
  } catch (error) {
    console.error('Error in addMinutesToTime:', error);
    return timeStr;
  }
};

/**
 * Calculate the duration in minutes between two time strings (HH:mm)
 */
export const calculateDuration = (startTime: string, endTime: string): number => {
  try {
    const [startHours, startMinutes] = startTime.split(':').map(Number);
    const [endHours, endMinutes] = endTime.split(':').map(Number);
    
    const startDate = new Date();
    startDate.setHours(startHours, startMinutes, 0, 0);
    
    let endDate = new Date();
    endDate.setHours(endHours, endMinutes, 0, 0);
    
    // If end time is before start time, assume it's the next day
    if (endDate <= startDate) {
      endDate = addDays(endDate, 1);
    }
    
    return (endDate.getTime() - startDate.getTime()) / (1000 * 60);
  } catch (error) {
    console.error('Error in calculateDuration:', error);
    return 0;
  }
};

/**
 * Format a duration in minutes as a human-readable string
 */
export const formatDuration = (minutes: number): string => {
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  
  if (hours === 0) {
    return `${mins}m`;
  }
  
  return `${hours}h ${mins > 0 ? `${mins}m` : ''}`.trim();
};
