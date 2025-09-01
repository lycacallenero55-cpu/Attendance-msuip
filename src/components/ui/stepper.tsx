import * as React from "react"
import { Check, Circle, Loader2 } from "lucide-react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

type StepStatus = "complete" | "current" | "upcoming"

const stepVariants = cva(
  "flex items-center justify-center rounded-full border-2 font-medium transition-colors",
  {
    variants: {
      status: {
        complete: "border-primary bg-primary text-primary-foreground",
        current: "border-primary bg-background text-foreground",
        upcoming: "border-muted-foreground/25 bg-background text-muted-foreground/50",
      },
      size: {
        sm: "h-6 w-6 text-xs",
        default: "h-8 w-8 text-sm",
        lg: "h-10 w-10 text-base",
      },
    },
    defaultVariants: {
      status: "upcoming",
      size: "default",
    },
  }
)

const stepLabelVariants = cva("text-sm font-medium transition-colors", {
  variants: {
    status: {
      complete: "text-foreground",
      current: "text-foreground font-semibold",
      upcoming: "text-muted-foreground",
    },
  },
  defaultVariants: {
    status: "upcoming",
  },
})

const stepDescriptionVariants = cva("text-sm transition-colors", {
  variants: {
    status: {
      complete: "text-muted-foreground",
      current: "text-foreground",
      upcoming: "text-muted-foreground/50",
    },
  },
  defaultVariants: {
    status: "upcoming",
  },
})

interface StepProps extends React.HTMLAttributes<HTMLDivElement> {
  status: StepStatus
  step: number
  label: string
  description?: string
  isLast?: boolean
  isLoading?: boolean
  size?: VariantProps<typeof stepVariants>["size"]
  className?: string
}

function Step({
  status,
  step,
  label,
  description,
  isLast = false,
  isLoading = false,
  size = "default",
  className,
  ...props
}: StepProps) {
  return (
    <div
      className={cn(
        "relative flex flex-1 flex-col items-center md:flex-row md:items-start",
        !isLast && "flex-1",
        className
      )}
      {...props}
    >
      <div className="flex flex-col items-center md:flex-row">
        <div className="relative">
          {isLoading ? (
            <div className={cn(stepVariants({ status, size }))}>
              <Loader2 className="h-4 w-4 animate-spin" />
            </div>
          ) : status === "complete" ? (
            <div className={cn(stepVariants({ status, size }))}>
              <Check className="h-4 w-4" />
            </div>
          ) : status === "current" ? (
            <div className={cn(stepVariants({ status, size }))}>
              <Circle className="h-2.5 w-2.5 fill-current text-current" />
            </div>
          ) : (
            <div className={cn(stepVariants({ status, size }))}>{step}</div>
          )}
        </div>
        <div className="mt-2 text-center md:ml-3 md:mt-0 md:text-left">
          <div className={cn(stepLabelVariants({ status }))}>
            {label}
          </div>
          {description && (
            <div className={cn(stepDescriptionVariants({ status }), "mt-1")}>
              {description}
            </div>
          )}
        </div>
      </div>
      {!isLast && (
        <div
          className={cn(
            "absolute top-4 left-1/2 -z-10 h-0.5 w-full -translate-y-1/2 md:left-1/2 md:top-1/2 md:h-0.5 md:w-full md:-translate-y-1/2 md:translate-x-1/2",
            status === "complete" ? "bg-primary" : "bg-muted-foreground/25"
          )}
          aria-hidden="true"
        />
      )}
    </div>
  )
}

interface StepperProps extends React.HTMLAttributes<HTMLDivElement> {
  activeStep: number
  steps: Array<{ label: string; description?: string; isLoading?: boolean }>
  orientation?: "horizontal" | "vertical"
  size?: VariantProps<typeof stepVariants>["size"]
  className?: string
}

function Stepper({
  activeStep,
  steps,
  orientation = "horizontal",
  size = "default",
  className,
  ...props
}: StepperProps) {
  return (
    <nav
      className={cn(
        "relative flex w-full flex-col md:flex-row md:items-start md:justify-between",
        orientation === "vertical" && "space-y-4",
        className
      )}
      aria-label="Progress"
      {...props}
    >
      {steps.map((step, index) => {
        const stepNumber = index + 1
        const stepStatus =
          stepNumber < activeStep
            ? "complete"
            : stepNumber === activeStep
              ? "current"
              : "upcoming"

        return (
          <Step
            key={index}
            status={stepStatus}
            step={stepNumber}
            label={step.label}
            description={step.description}
            isLast={index === steps.length - 1}
            isLoading={step.isLoading}
            size={size}
          />
        )
      })}
    </nav>
  )
}

export { Stepper, Step, type StepStatus }
