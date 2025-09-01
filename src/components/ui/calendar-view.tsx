import * as React from "react"
import { format, startOfWeek, addDays, isSameDay, isSameMonth, isToday, parseISO, endOfWeek, startOfMonth, endOfMonth, eachDayOfInterval, isWithinInterval } from "date-fns"
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react"

import { Button } from "./button"
import { cn } from "@/lib/utils"

export interface CalendarEvent {
  date: Date | string;
  title: string;
  type?: 'default' | 'holiday' | 'event' | 'attendance';
  className?: string;
}

interface CalendarViewProps {
  /**
   * The current date to display in the calendar
   * @default new Date()
   */
  date?: Date
  
  /**
   * Callback when a date is selected
   */
  onDateSelect?: (date: Date) => void
  
  /**
   * Callback when the month changes
   */
  onMonthChange?: (date: Date) => void
  
  /**
   * Array of dates to mark as events
   */
  events?: Array<{
    date: Date | string
    title: string
    type?: 'default' | 'holiday' | 'event' | 'attendance'
    className?: string
  }>
  
  /**
   * Custom class name for the calendar container
   */
  className?: string
  
  /**
   * Show/hide navigation controls
   * @default true
   */
  showNavigation?: boolean
  
  /**
   * Show/hide the header with month/year and navigation
   * @default true
   */
  showHeader?: boolean
  
  /**
   * Show/hide the day names row
   * @default true
   */
  showDayNames?: boolean
  
  /**
   * Show/hide the week numbers
   * @default false
   */
  showWeekNumbers?: boolean
  
  /**
   * The first day of the week (0 = Sunday, 1 = Monday, etc.)
   * @default 0 (Sunday)
   */
  firstDayOfWeek?: number
  
  /**
   * Custom renderer for day cells
   */
  renderDay?: (date: Date, events: CalendarEvent[]) => React.ReactNode
  
  /**
   * Custom renderer for event indicators
   */
  renderEvent?: (event: CalendarEvent, date: Date) => React.ReactNode
  
  /**
   * Custom class names for different parts of the calendar
   */
  classNames?: {
    header?: string
    navigation?: string
    dayNames?: string
    week?: string
    day?: string
    today?: string
    selected?: string
    outsideMonth?: string
    event?: string
    eventType?: Record<string, string>
  }
}

const defaultEventTypes = {
  default: 'bg-blue-100 text-blue-800',
  holiday: 'bg-red-100 text-red-800',
  event: 'bg-purple-100 text-purple-800',
  attendance: 'bg-green-100 text-green-800',
}

export function CalendarView({
  date: currentDate = new Date(),
  onDateSelect,
  onMonthChange,
  events = [],
  className,
  showNavigation = true,
  showHeader = true,
  showDayNames = true,
  showWeekNumbers = false,
  firstDayOfWeek = 0,
  renderDay,
  renderEvent,
  classNames = {},
}: CalendarViewProps) {
  const [date, setDate] = React.useState<Date>(currentDate)
  const [selectedDate, setSelectedDate] = React.useState<Date | null>(null)

  // Update internal date when prop changes
  React.useEffect(() => {
    setDate(currentDate)
  }, [currentDate])

  // Process events to be more efficient
  const processedEvents = React.useMemo(() => {
    return events.map(event => ({
      ...event,
      date: typeof event.date === 'string' ? parseISO(event.date) : event.date,
    }))
  }, [events])

  // Get the start and end of the current month
  const monthStart = startOfMonth(date)
  const monthEnd = endOfMonth(date)
  
  // Get the start and end of the week based on firstDayOfWeek
  const startDate = startOfWeek(monthStart, { weekStartsOn: firstDayOfWeek as 0 | 1 | 2 | 3 | 4 | 5 | 6 })
  const endDate = endOfWeek(monthEnd, { weekStartsOn: firstDayOfWeek as 0 | 1 | 2 | 3 | 4 | 5 | 6 })

  // Generate all days in the current view
  const daysInMonth = eachDayOfInterval({ start: startDate, end: endDate })
  
  // Group days by week
  const weeks: Date[][] = []
  let currentWeek: Date[] = []
  
  daysInMonth.forEach((day, index) => {
    currentWeek.push(day)
    
    // Start a new week after 7 days or at the end of the array
    if ((index + 1) % 7 === 0 || index === daysInMonth.length - 1) {
      weeks.push([...currentWeek])
      currentWeek = []
    }
  })

  // Navigate to previous/next month
  const previousMonth = () => {
    const newDate = new Date(date.getFullYear(), date.getMonth() - 1, 1)
    setDate(newDate)
    onMonthChange?.(newDate)
  }

  const nextMonth = () => {
    const newDate = new Date(date.getFullYear(), date.getMonth() + 1, 1)
    setDate(newDate)
    onMonthChange?.(newDate)
  }

  const previousYear = () => {
    const newDate = new Date(date.getFullYear() - 1, date.getMonth(), 1)
    setDate(newDate)
    onMonthChange?.(newDate)
  }

  const nextYear = () => {
    const newDate = new Date(date.getFullYear() + 1, date.getMonth(), 1)
    setDate(newDate)
    onMonthChange?.(newDate)
  }

  const goToToday = () => {
    const today = new Date()
    setDate(today)
    onMonthChange?.(today)
  }

  // Handle date selection
  const handleDateSelect = (day: Date) => {
    setSelectedDate(day)
    onDateSelect?.(day)
  }

  // Get events for a specific day
  const getEventsForDay = (day: Date) => {
    return processedEvents.filter(event => isSameDay(event.date, day))
  }

  interface CalendarEvent {
    date: Date | string;
    title: string;
    type?: 'default' | 'holiday' | 'event' | 'attendance';
    className?: string;
  }

  // Default day cell renderer
  const defaultRenderDay = (day: Date, dayEvents: CalendarEvent[]) => {
    const isSelected = selectedDate ? isSameDay(day, selectedDate) : false
    const isCurrentDay = isToday(day)
    const isCurrentMonth = isSameMonth(day, date)
    const dayEventsCount = dayEvents.length
    const hasEvents = dayEventsCount > 0
    
    return (
      <div 
        className={cn(
          "relative flex h-8 w-8 items-center justify-center rounded-full text-sm transition-colors",
          isCurrentDay && !isSelected && "border border-blue-500 font-medium",
          isSelected && "bg-blue-500 text-white",
          !isCurrentMonth && "text-muted-foreground/50",
          hasEvents && "font-medium",
          classNames.day,
          isCurrentDay && classNames.today,
          isSelected && classNames.selected,
          !isCurrentMonth && classNames.outsideMonth,
        )}
        onClick={() => handleDateSelect(day)}
      >
        {format(day, 'd')}
        {hasEvents && (
          <div className="absolute -bottom-1 left-1/2 h-1 w-1 -translate-x-1/2 transform rounded-full bg-blue-500"></div>
        )}
      </div>
    )
  }

  // Default event renderer
  const defaultRenderEvent = (event: CalendarEvent, day: Date) => {
    const eventType = event.type || 'default'
    const typeClass = classNames.eventType?.[eventType] || defaultEventTypes[eventType as keyof typeof defaultEventTypes] || defaultEventTypes.default
    
    return (
      <div 
        key={`${event.date}-${event.title}`}
        className={cn(
          "mb-1 truncate rounded px-1 text-xs",
          typeClass,
          classNames.event,
          event.className
        )}
        title={event.title}
      >
        {event.title}
      </div>
    )
  }

  // Day names for the header
  const dayNames = Array.from({ length: 7 }).map((_, i) => {
    const day = addDays(startDate, i)
    return format(day, 'EEE')
  })

  return (
    <div className={cn("w-full max-w-3xl space-y-4", className)}>
      {showHeader && (
        <div className={cn("flex items-center justify-between", classNames.header)}>
          <div className="flex items-center space-x-2">
            <h2 className="text-xl font-semibold">
              {format(date, 'MMMM yyyy')}
            </h2>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={goToToday}
              className="text-sm text-muted-foreground hover:text-foreground"
            >
              Today
            </Button>
          </div>
          
          {showNavigation && (
            <div className={cn("flex items-center space-x-1", classNames.navigation)}>
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8"
                onClick={previousYear}
              >
                <ChevronsLeft className="h-4 w-4" />
                <span className="sr-only">Previous year</span>
              </Button>
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8"
                onClick={previousMonth}
              >
                <ChevronLeft className="h-4 w-4" />
                <span className="sr-only">Previous month</span>
              </Button>
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8"
                onClick={nextMonth}
              >
                <ChevronRight className="h-4 w-4" />
                <span className="sr-only">Next month</span>
              </Button>
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8"
                onClick={nextYear}
              >
                <ChevronsRight className="h-4 w-4" />
                <span className="sr-only">Next year</span>
              </Button>
            </div>
          )}
        </div>
      )}
      
      <div className="space-y-2">
        {showDayNames && (
          <div className={cn("grid grid-cols-7 text-center text-sm font-medium text-muted-foreground", classNames.dayNames)}>
            {dayNames.map((day, index) => (
              <div key={index} className="py-2">
                {day}
              </div>
            ))}
          </div>
        )}
        
        <div className="space-y-1">
          {weeks.map((week, weekIndex) => (
            <div 
              key={weekIndex} 
              className={cn("grid grid-cols-7 text-center", classNames.week)}
            >
              {showWeekNumbers && (
                <div className="flex items-center justify-center text-xs text-muted-foreground">
                  {format(week[0], 'w')}
                </div>
              )}
              
              {week.map((day, dayIndex) => {
                const dayEvents = getEventsForDay(day)
                const isCurrentMonth = isSameMonth(day, date)
                
                return (
                  <div 
                    key={dayIndex} 
                    className={cn(
                      "relative min-h-24 border p-1 text-left",
                      !isCurrentMonth && "bg-muted/20",
                      isToday(day) && "bg-blue-50 dark:bg-blue-900/20",
                      selectedDate && isSameDay(day, selectedDate) && "ring-2 ring-blue-500"
                    )}
                    onClick={() => handleDateSelect(day)}
                  >
                    <div className="mb-1">
                      {renderDay 
                        ? renderDay(day, dayEvents)
                        : defaultRenderDay(day, dayEvents)
                      }
                    </div>
                    
                    <div className="space-y-1 overflow-hidden">
                      {dayEvents.map(event => 
                        renderEvent 
                          ? renderEvent(event, day)
                          : defaultRenderEvent(event, day)
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Export a compact version of the calendar for use in date pickers
export function MiniCalendar({
  date = new Date(),
  onDateSelect,
  className,
}: Pick<CalendarViewProps, 'date' | 'onDateSelect' | 'className'>) {
  return (
    <CalendarView
      date={date}
      onDateSelect={onDateSelect}
      className={cn("w-64", className)}
      showNavigation={false}
      showHeader={false}
      showDayNames={true}
      classNames={{
        day: "h-6 w-6 text-xs",
        today: "bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200",
        selected: "bg-blue-500 text-white hover:bg-blue-600",
      }}
    />
  )
}
