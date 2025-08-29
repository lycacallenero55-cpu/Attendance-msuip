import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"
import { Check, Clock, X, AlertCircle, MoreHorizontal } from "lucide-react"

const timelineVariants = cva("relative pl-6 border-l-2", {
  variants: {
    variant: {
      default: "border-border",
      primary: "border-primary",
      success: "border-emerald-500",
      warning: "border-amber-500",
      danger: "border-red-500",
    },
  },
  defaultVariants: {
    variant: "default",
  },
})

const dotVariants = cva(
  "absolute -left-2.5 top-0 flex h-5 w-5 items-center justify-center rounded-full border-2 ring-4 ring-background",
  {
    variants: {
      variant: {
        default: "bg-background border-border",
        primary: "bg-primary border-primary text-primary-foreground",
        success: "bg-emerald-500 border-emerald-500 text-white",
        warning: "bg-amber-500 border-amber-500 text-white",
        danger: "bg-red-500 border-red-500 text-white",
      },
      icon: {
        true: "p-0.5",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

interface TimelineProps extends React.HTMLAttributes<HTMLUListElement> {
  variant?: VariantProps<typeof timelineVariants>["variant"]
}

function Timeline({ className, variant, ...props }: TimelineProps) {
  return (
    <ul
      className={cn(timelineVariants({ variant }), "space-y-8", className)}
      {...props}
    />
  )
}

interface TimelineItemProps extends React.LiHTMLAttributes<HTMLLIElement> {
  dot?: React.ReactNode
  variant?: VariantProps<typeof dotVariants>["variant"]
}

function TimelineItem({
  className,
  children,
  dot,
  variant = "default",
  ...props
}: TimelineItemProps) {
  return (
    <li className={cn("relative pl-6", className)} {...props}>
      <div className={cn(dotVariants({ variant, icon: !!dot }), "-left-2.5")}>
        {dot}
      </div>
      <div className="pb-8">{children}</div>
    </li>
  )
}

interface TimelineHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  time?: string
  action?: React.ReactNode
}

function TimelineHeader({
  className,
  children,
  time,
  action,
  ...props
}: TimelineHeaderProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-4 text-sm font-medium leading-none",
        className
      )}
      {...props}
    >
      <div>{children}</div>
      <div className="flex items-center gap-2">
        {time && <span className="text-xs text-muted-foreground">{time}</span>}
        {action}
      </div>
    </div>
  )
}

function TimelineContent({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("mt-1 text-sm text-muted-foreground", className)}
      {...props}
    />
  )
}

// Predefined status items for common use cases
function TimelineStatus({
  status = "default",
  className,
  ...props
}: {
  status: "default" | "success" | "warning" | "error" | "pending"
} & React.HTMLAttributes<HTMLDivElement>) {
  const statusConfig = {
    default: {
      icon: MoreHorizontal,
      className: "text-muted-foreground",
      label: "In Progress",
    },
    success: {
      icon: Check,
      className: "text-emerald-600 dark:text-emerald-400",
      label: "Completed",
    },
    warning: {
      icon: Clock,
      className: "text-amber-600 dark:text-amber-400",
      label: "Pending",
    },
    error: {
      icon: X,
      className: "text-red-600 dark:text-red-400",
      label: "Failed",
    },
    pending: {
      icon: Clock,
      className: "text-blue-600 dark:text-blue-400",
      label: "Pending",
    },
  }[status]

  const Icon = statusConfig.icon

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
        statusConfig.className,
        className
      )}
      {...props}
    >
      <Icon className="h-3 w-3" />
      {statusConfig.label}
    </div>
  )
}

export {
  Timeline,
  TimelineItem,
  TimelineHeader,
  TimelineContent,
  TimelineStatus,
  timelineVariants,
  dotVariants,
}

// Example usage:
/*
<Timeline>
  <TimelineItem>
    <TimelineHeader time="9:30 AM">
      Check-in
      <TimelineStatus status="success" />
    </TimelineHeader>
    <TimelineContent>Employee checked in at the main office.</TimelineContent>
  </TimelineItem>
  <TimelineItem variant="warning">
    <TimelineHeader time="1:15 PM">
      Lunch Break
      <TimelineStatus status="warning" />
    </TimelineHeader>
    <TimelineContent>Currently on lunch break.</TimelineContent>
  </TimelineItem>
</Timeline>
*/
