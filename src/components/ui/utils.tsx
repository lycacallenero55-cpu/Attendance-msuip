import * as React from 'react';
import { Check, Copy, Loader2, QrCode, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from './button';
import { toast } from '@/components/ui/use-toast';
import { VisuallyHidden } from '@/lib/accessibility';

// QR Code component using canvas
interface QRCodeCanvasProps extends React.CanvasHTMLAttributes<HTMLCanvasElement> {
  value: string;
  width?: number;
  height?: number;
  margin?: number;
  errorCorrectionLevel?: 'L' | 'M' | 'Q' | 'H';
}

const QRCodeCanvas = React.lazy(() => 
  import('qrcode').then(QRCode => {
    const QRCodeComponent: React.FC<QRCodeCanvasProps> = ({
      value,
      width = 256,
      height = 256,
      margin = 2,
      errorCorrectionLevel = 'M',
      ...props
    }) => {
      const canvasRef = React.useRef<HTMLCanvasElement>(null);
      
      React.useEffect(() => {
        if (!canvasRef.current || !value) return;
        
        // Create a copy of the options without the errorCorrectionLevel
        const qrOptions = {
          width,
          margin,
          errorCorrectionLevel,
          color: {
            dark: '#000000',
            light: '#ffffff'
          }
        };
        
  
        QRCode.toCanvas(canvasRef.current, value, {
          width: 256,
          margin: 2,
          errorCorrectionLevel: 'H',
          color: {
            dark: '#000000',
            light: '#ffffff'
          }
        })
          .catch(console.error);
      }, [value, width, height, margin, errorCorrectionLevel]);
      
      return (
        <canvas 
          ref={canvasRef} 
          width={width} 
          height={height} 
          style={{ width: '100%', height: 'auto' }}
          {...props} 
        />
      );
    };
    
    return { default: QRCodeComponent };
  })
);

// Copy to Clipboard Button
interface CopyButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  value: string;
  label?: string;
  onCopy?: () => void;
  size?: 'sm' | 'default' | 'lg' | 'icon';
  variant?: 'ghost' | 'outline' | 'link' | 'default' | 'destructive' | 'secondary';
}

export function CopyButton({
  value,
  label = 'Copy',
  onCopy,
  className,
  size = 'icon',
  variant = 'ghost',
  ...props
}: CopyButtonProps) {
  const [isCopied, setIsCopied] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(false);
  
  const handleCopy = async () => {
    if (isLoading) return;
    
    try {
      setIsLoading(true);
      
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(value);
      } else {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = value;
        textarea.style.position = 'fixed';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      }
      
      setIsCopied(true);
      onCopy?.();
      
      toast({
        title: 'Success',
        description: 'Copied to clipboard',
        variant: 'default',
      });
      
      // Reset after 2 seconds
      setTimeout(() => {
        setIsCopied(false);
      }, 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
      toast({
        title: 'Error',
        description: 'Failed to copy to clipboard',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <Button
      type="button"
      variant={variant}
      size={size}
      onClick={handleCopy}
      disabled={isLoading}
      className={cn('relative', className)}
      {...props}
    >
      {isLoading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : isCopied ? (
        <Check className="h-4 w-4" />
      ) : (
        <Copy className="h-4 w-4" />
      )}
      {label && size !== 'icon' && (
        <span className="ml-2">
          {isCopied ? 'Copied!' : label}
        </span>
      )}
      <VisuallyHidden>Copy to clipboard</VisuallyHidden>
    </Button>
  );
}

// QR Code Component
interface QRCodeProps extends React.HTMLAttributes<HTMLDivElement> {
  value: string;
  size?: number;
  level?: 'L' | 'M' | 'Q' | 'H';
  includeMargin?: boolean;
  renderAs?: 'canvas' | 'svg';
  className?: string;
  buttonClassName?: string;
  buttonVariant?: 'ghost' | 'outline' | 'link' | 'default' | 'destructive' | 'secondary';
  buttonSize?: 'sm' | 'default' | 'lg' | 'icon';
  showCopyButton?: boolean;
  copyButtonLabel?: string;
  onCopy?: () => void;
}

export function QRCode({
  value,
  size = 128,
  level = 'M',
  includeMargin = false,
  renderAs = 'canvas',
  className,
  buttonClassName,
  buttonVariant = 'outline',
  buttonSize = 'icon',
  showCopyButton = true,
  copyButtonLabel = 'Copy',
  onCopy,
  ...props
}: QRCodeProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [isMounted, setIsMounted] = React.useState(false);
  
  React.useEffect(() => {
    setIsMounted(true);
  }, []);
  
  if (!isMounted) {
    return (
      <div 
        className={cn(
          'flex items-center justify-center bg-muted rounded-md',
          `w-[${size}px] h-[${size}px]`,
          className
        )}
        {...props}
      >
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }
  
  return (
    <div className={cn('flex flex-col items-center gap-2', className)} {...props}>
      <div className="relative group">
        <div 
          className={cn(
            'rounded-md overflow-hidden border border-border',
            'flex items-center justify-center',
            'transition-all duration-200',
            'hover:shadow-md',
            `w-[${size}px] h-[${size}px]`
          )}
          onClick={() => setIsOpen(true)}
        >
          <React.Suspense
            fallback={
              <div className="flex items-center justify-center w-full h-full bg-muted">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            }
          >
            <QRCodeCanvas 
              value={value}
              width={size}
              height={size}
              margin={includeMargin ? 2 : 0}
              errorCorrectionLevel={level}
              className="w-full h-full"
            />
          </React.Suspense>
        </div>
        
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-black/50 transition-opacity rounded-md">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            className="text-white bg-black/50 hover:bg-black/70"
            onClick={(e) => {
              e.stopPropagation();
              setIsOpen(true);
            }}
          >
            <QrCode className="h-4 w-4 mr-1" />
            <span>Enlarge</span>
          </Button>
        </div>
      </div>
      
      {showCopyButton && (
        <CopyButton
          value={value}
          label={copyButtonLabel}
          onCopy={onCopy}
          variant={buttonVariant}
          size={buttonSize}
          className={buttonClassName}
        />
      )}
      
      {/* QR Code Modal */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80">
          <div className="relative bg-background rounded-lg p-6 max-w-md w-full">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">QR Code</h3>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => setIsOpen(false)}
                className="h-8 w-8"
              >
                <X className="h-4 w-4" />
                <VisuallyHidden>Close</VisuallyHidden>
              </Button>
            </div>
            
            <div className="flex flex-col items-center gap-4">
              <div className="p-4 bg-white rounded-md">
                <React.Suspense
                  fallback={
                    <div className="flex items-center justify-center w-64 h-64 bg-muted">
                      <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                    </div>
                  }
                >
                  <QRCodeCanvas 
                    value={value}
                    width={256}
                    height={256}
                    margin={2}
                    errorCorrectionLevel="H"
                  />
                </React.Suspense>
              </div>
              
              <div className="w-full flex flex-col sm:flex-row gap-2">
                <div className="flex-1 p-2 bg-muted rounded-md text-sm break-all">
                  {value}
                </div>
                <CopyButton
                  value={value}
                  label="Copy"
                  onCopy={onCopy}
                  variant="default"
                  size="sm"
                  className="shrink-0"
                />
              </div>
              
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = `data:image/svg+xml;utf8,${encodeURIComponent(
                      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">${document.querySelector('canvas')?.toDataURL('image/svg+xml')}</svg>`
                    )}`;
                    link.download = 'qrcode.svg';
                    link.click();
                  }}
                >
                  Download SVG
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = document.querySelector('canvas')?.toDataURL('image/png') || '';
                    link.download = 'qrcode.png';
                    link.click();
                  }}
                >
                  Download PNG
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Signature Pad Component
interface SignaturePadProps extends React.HTMLAttributes<HTMLDivElement> {
  onSave?: (signature: string) => void;
  onClear?: () => void;
  width?: number | string;
  height?: number | string;
  penColor?: string;
  backgroundColor?: string;
  clearButtonText?: string;
  saveButtonText?: string;
  showButtons?: boolean;
  className?: string;
  buttonClassName?: string;
  buttonVariant?: 'ghost' | 'outline' | 'link' | 'default' | 'destructive' | 'secondary';
  buttonSize?: 'sm' | 'default' | 'lg';
  disabled?: boolean;
}

export function SignaturePad({
  onSave,
  onClear,
  width = '100%',
  height = 200,
  penColor = '#000000',
  backgroundColor = 'transparent',
  clearButtonText = 'Clear',
  saveButtonText = 'Save',
  showButtons = true,
  className,
  buttonClassName,
  buttonVariant = 'outline',
  buttonSize = 'default',
  disabled = false,
  ...props
}: SignaturePadProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = React.useState(false);
  const [lastX, setLastX] = React.useState(0);
  const [lastY, setLastY] = React.useState(0);
  const [hasSignature, setHasSignature] = React.useState(false);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setIsDrawing(true);
    setLastX(x);
    setLastY(y);
  };

  const handleTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (disabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const y = e.touches[0].clientY - rect.top;

    setIsDrawing(true);
    setLastX(x);
    setLastY(y);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || disabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.strokeStyle = penColor;
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    setLastX(x);
    setLastY(y);
    setHasSignature(true);
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing || disabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const y = e.touches[0].clientY - rect.top;

    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.strokeStyle = penColor;
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    setLastX(x);
    setLastY(y);
    setHasSignature(true);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clear = React.useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setHasSignature(false);
    onClear?.();
  }, [onClear]);

  const save = React.useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dataUrl = canvas.toDataURL('image/png');
    onSave?.(dataUrl);
  }, [onSave]);

  // Initialize canvas
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to handle high DPI displays
    const setCanvasSize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();

      // Set display size (css pixels)
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;

      // Set actual size in memory (scaled for DPI)
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      // Scale context to ensure correct drawing operations
      ctx.scale(dpr, dpr);

      // Set initial styles
      ctx.strokeStyle = penColor;
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
    };

    setCanvasSize();

    // Handle window resize
    const handleResize = () => {
      setCanvasSize();
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };

    ctx.strokeStyle = penColor;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
  }, [penColor]);

  return (
    <div className={cn('flex flex-col gap-2', className)} {...props}>
      <div
        className={cn(
          'relative border rounded-md overflow-hidden',
          'bg-white dark:bg-gray-900',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
        style={{ width, height }}
      >
        <canvas
          ref={canvasRef}
          className={cn(
            'w-full h-full touch-none',
            disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-crosshair'
          )}
          style={{
            backgroundColor,
            width,
            height,
            ...props.style,
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={stopDrawing}
        />
        {!hasSignature && !disabled && (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground pointer-events-none">
            <span className="text-sm">Sign here</span>
          </div>
        )}
      </div>

      {showButtons && (
        <div className="flex gap-2">
          <Button
            type="button"
            variant={buttonVariant}
            size={buttonSize}
            onClick={clear}
            disabled={!hasSignature || disabled}
            className={buttonClassName}
          >
            {clearButtonText}
          </Button>
          
          <Button
            type="button"
            variant={buttonVariant === 'outline' ? 'default' : 'outline'}
            size={buttonSize}
            onClick={save}
            disabled={!hasSignature || disabled}
            className={cn('ml-auto', buttonClassName)}
          >
            {saveButtonText}
          </Button>
        </div>
      )}
    </div>
  );
}

// Export all components
export default {
  CopyButton,
  QRCode,
  SignaturePad,
};
