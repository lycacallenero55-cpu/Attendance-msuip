import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

export default function ErrorFallback({ 
  error, 
  resetErrorBoundary 
}: ErrorFallbackProps) {
  return (
    <div className="p-4">
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Something went wrong</AlertTitle>
        <AlertDescription>
          <p className="mb-2">{error.message}</p>
          <Button 
            size="sm" 
            onClick={resetErrorBoundary}
          >
            Try Again
          </Button>
        </AlertDescription>
      </Alert>
    </div>
  );
}
