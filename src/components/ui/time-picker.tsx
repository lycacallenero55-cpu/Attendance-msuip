import * as React from "react"
import { format } from "date-fns"
import { Clock } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Input } from "@/components/ui/input"

interface TimePickerProps {
  date?: Date
  onSelect: (date: Date | undefined) => void
  className?: string
  disabled?: boolean
}

export function TimePicker({
  date,
  onSelect,
  className,
  disabled = false,
}: TimePickerProps) {
  const [open, setOpen] = React.useState(false)
  const [inputValue, setInputValue] = React.useState(
    date ? format(date, "HH:mm") : ""
  )

  const handleTimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setInputValue(value)
    
    // Validate time format (HH:MM)
    if (/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/.test(value)) {
      const [hours, minutes] = value.split(":").map(Number)
      const newDate = date ? new Date(date) : new Date()
      newDate.setHours(hours, minutes, 0, 0)
      onSelect(newDate)
    }
  }

  const handleSelectNow = () => {
    const now = new Date()
    setInputValue(format(now, "HH:mm"))
    onSelect(now)
    setOpen(false)
  }

  const handleClear = () => {
    setInputValue("")
    onSelect(undefined)
    setOpen(false)
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start text-left font-normal",
            !date && "text-muted-foreground",
            className
          )}
          disabled={disabled}
        >
          <Clock className="mr-2 h-4 w-4" />
          {date ? format(date, "h:mm a") : <span>Pick a time</span>}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <div className="p-4">
          <div className="mb-4">
            <Input
              type="time"
              value={inputValue}
              onChange={handleTimeChange}
              className="w-full"
              step={300} // 5 minute steps
            />
          </div>
          <div className="flex justify-between">
            <Button
              variant="outline"
              size="sm"
              onClick={handleSelectNow}
            >
              Now
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClear}
              disabled={!date}
            >
              Clear
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}

interface TimeRangePickerProps {
  startTime?: Date
  endTime?: Date
  onStartTimeSelect: (date: Date | undefined) => void
  onEndTimeSelect: (date: Date | undefined) => void
  className?: string
  disabled?: boolean
}

export function TimeRangePicker({
  startTime,
  endTime,
  onStartTimeSelect,
  onEndTimeSelect,
  className,
  disabled = false,
}: TimeRangePickerProps) {
  return (
    <div className={cn("flex items-center space-x-2", className)}>
      <TimePicker
        date={startTime}
        onSelect={onStartTimeSelect}
        disabled={disabled}
      />
      <span className="mx-1 text-muted-foreground">to</span>
      <TimePicker
        date={endTime}
        onSelect={onEndTimeSelect}
        disabled={disabled || !startTime}
      />
    </div>
  )
}
