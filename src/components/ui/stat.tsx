import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"
import { ArrowDown, ArrowUp, ArrowRight, Loader2 } from "lucide-react"

const statVariants = cva(
  "rounded-lg border p-4 shadow-sm transition-colors",
  {
    variants: {
      variant: {
        default: "bg-card text-card-foreground",
        primary: "bg-primary/5 border-primary/20 text-primary-foreground",
        secondary: "bg-secondary/5 border-secondary/20 text-secondary-foreground",
        success: "bg-emerald-50 border-emerald-100 text-emerald-800 dark:bg-emerald-950/20 dark:border-emerald-900/50 dark:text-emerald-400",
        warning: "bg-amber-50 border-amber-100 text-amber-800 dark:bg-amber-950/20 dark:border-amber-900/50 dark:text-amber-400",
        danger: "bg-red-50 border-red-100 text-red-800 dark:bg-red-950/20 dark:border-red-900/50 dark:text-red-400",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

const iconVariants = cva(
  "flex h-10 w-10 items-center justify-center rounded-full",
  {
    variants: {
      variant: {
        default: "bg-muted text-muted-foreground",
        primary: "bg-primary/10 text-primary",
        secondary: "bg-secondary/10 text-secondary-foreground",
        success: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400",
        warning: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
        danger: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

const trendVariants = cva("inline-flex items-center text-sm font-medium", {
  variants: {
    trend: {
      up: "text-emerald-600 dark:text-emerald-400",
      down: "text-red-600 dark:text-red-400",
      neutral: "text-muted-foreground",
    },
  },
  defaultVariants: {
    trend: "neutral",
  },
})

interface StatProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof statVariants> {
  title: string
  value: string | number
  description?: string
  icon?: React.ReactNode
  trend?: {
    value: number
    label: string
  }
  loading?: boolean
}

export function Stat({
  className,
  variant,
  title,
  value,
  description,
  icon,
  trend,
  loading = false,
  ...props
}: StatProps) {
  const TrendIcon = React.useMemo(() => {
    if (!trend) return null
    if (trend.value > 0) return ArrowUp
    if (trend.value < 0) return ArrowDown
    return ArrowRight
  }, [trend])

  const trendValue = React.useMemo(() => {
    if (!trend) return null
    return Math.abs(trend.value)
  }, [trend])

  const trendType = React.useMemo(() => {
    if (!trend) return "neutral"
    if (trend.value > 0) return "up"
    if (trend.value < 0) return "down"
    return "neutral"
  }, [trend])

  return (
    <div className={cn(statVariants({ variant }), className)} {...props}>
      <div className="flex items-center justify-between">
        <div className="space-y-0.5">
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          {loading ? (
            <div className="flex items-center space-x-2">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <span className="sr-only">Loading...</span>
            </div>
          ) : (
            <h3 className="text-2xl font-bold tracking-tight">{value}</h3>
          )}
          {description && (
            <p className="text-xs text-muted-foreground">{description}</p>
          )}
          {trend && TrendIcon && trendValue !== null && (
            <div className={cn(trendVariants({ trend: trendType }))}>
              <TrendIcon className="mr-1 h-3.5 w-3.5" />
              {trendValue}% {trend.label}
            </div>
          )}
        </div>
        {icon && <div className={cn(iconVariants({ variant }))}>{icon}</div>}
      </div>
    </div>
  )
}

interface StatGridProps extends React.HTMLAttributes<HTMLDivElement> {
  cols?: 1 | 2 | 3 | 4 | 5 | 6
  gap?: "sm" | "md" | "lg"
}

export function StatGrid({
  className,
  cols = 3,
  gap = "md",
  children,
  ...props
}: StatGridProps) {
  const gapClasses = {
    sm: "gap-2",
    md: "gap-4",
    lg: "gap-6",
  }

  const gridCols = {
    1: "grid-cols-1",
    2: "grid-cols-1 sm:grid-cols-2",
    3: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4",
    5: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-5",
    6: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6",
  }

  return (
    <div
      className={cn(
        "grid w-full",
        gridCols[cols],
        gapClasses[gap],
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}
