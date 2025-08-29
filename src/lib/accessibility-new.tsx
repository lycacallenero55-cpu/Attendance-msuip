import * as React from 'react';
import { useCallback, useEffect, useRef, useState } from 'react';

// Utility function to safely concatenate class names
export function cn(...classes: (string | undefined | null | false)[]): string {
  return classes.filter(Boolean).join(' ');
}

// VisuallyHidden component for screen reader only content
export const VisuallyHidden: React.FC<React.HTMLAttributes<HTMLSpanElement>> = ({
  children,
  className = '',
  ...props
}) => {
  const [forceShow, setForceShow] = useState(false);
  const isDev = process.env.NODE_ENV === 'development';

  useEffect(() => {
    if (!isDev) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Alt') {
        setForceShow(true);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Alt') {
        setForceShow(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isDev]);

  if (forceShow) {
    return (
      <span className="fixed bottom-0 left-0 z-50 m-4 rounded bg-black p-2 text-white">
        {children}
      </span>
    );
  }

  return (
    <span
      className={cn('sr-only', className)}
      {...props}
    >
      {children}
    </span>
  );
};

// SkipLink component for keyboard navigation
export const SkipLink: React.FC<React.AnchorHTMLAttributes<HTMLAnchorElement>> = ({
  children = 'Skip to content',
  className = '',
  ...props
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLAnchorElement>(null);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Tab') {
      setIsVisible(true);
    }
  }, []);

  const handleClick = useCallback(() => {
    setIsVisible(false);
  }, []);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('click', handleClick);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('click', handleClick);
    };
  }, [handleClick, handleKeyDown]);

  if (!isVisible) return null;

  return (
    <a
      ref={ref}
      href="#main"
      onClick={handleClick}
      className={cn(
        'sr-only',
        'focus:not-sr-only',
        'focus:fixed focus:top-4 focus:left-4 focus:z-50',
        'focus:bg-white focus:px-4 focus:py-2 focus:font-bold focus:text-black',
        'focus:outline focus:outline-2 focus:outline-offset-2 focus:outline-blue-500',
        className
      )}
      {...props}
    >
      {children}
    </a>
  );
};

// FocusTrap component for keyboard focus management
interface FocusTrapProps extends React.HTMLAttributes<HTMLDivElement> {
  active?: boolean;
  returnFocusOnDeactivate?: boolean;
  initialFocus?: HTMLElement | null;
}

export const FocusTrap: React.FC<FocusTrapProps> = ({
  children,
  active = true,
  returnFocusOnDeactivate = true,
  initialFocus = null,
  ...props
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const lastFocusedElement = useRef<Element | null>(null);
  const firstFocusableElement = useRef<HTMLElement | null>(null);
  const lastFocusableElement = useRef<HTMLElement | null>(null);

  const getFocusableElements = useCallback((): HTMLElement[] => {
    if (!containerRef.current) return [];
    
    return Array.from(
      containerRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )
    ).filter((el): el is HTMLElement => {
      return el instanceof HTMLElement && 
             !el.hasAttribute('disabled') && 
             el.getAttribute('aria-hidden') !== 'true';
    });
  }, []);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;

    const focusableElements = getFocusableElements();
    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      }
    } else {
      if (document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    }
  }, [getFocusableElements]);

  useEffect(() => {
    if (!active) return;

    // Save the currently focused element to return focus later
    lastFocusedElement.current = document.activeElement;

    // Get all focusable elements
    const focusableElements = getFocusableElements();
    
    if (focusableElements.length > 0) {
      firstFocusableElement.current = focusableElements[0];
      lastFocusableElement.current = focusableElements[focusableElements.length - 1];
      
      // Focus the initial element if provided, otherwise focus the first focusable element
      if (initialFocus && initialFocus instanceof HTMLElement) {
        initialFocus.focus();
      } else {
        firstFocusableElement.current.focus();
      }
    }

    // Add event listener for keyboard navigation
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      // Clean up event listener
      document.removeEventListener('keydown', handleKeyDown);

      // Return focus to the previously focused element if needed
      if (returnFocusOnDeactivate && lastFocusedElement.current instanceof HTMLElement) {
        (lastFocusedElement.current as HTMLElement).focus();
      }
    };
  }, [active, getFocusableElements, handleKeyDown, initialFocus, returnFocusOnDeactivate]);

  if (!active) return <>{children}</>;

  return (
    <div ref={containerRef} {...props}>
      {children}
    </div>
  );
};

// useFocusTrap hook for programmatic focus management
export function useFocusTrap(active = true) {
  const containerRef = useRef<HTMLElement | null>(null);
  const lastFocusedElement = useRef<Element | null>(null);
  const firstFocusableElement = useRef<HTMLElement | null>(null);
  const lastFocusableElement = useRef<HTMLElement | null>(null);

  const getFocusableElements = useCallback((): HTMLElement[] => {
    if (!containerRef.current) return [];
    
    return Array.from(
      containerRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )
    ).filter((el): el is HTMLElement => {
      return el instanceof HTMLElement && 
             !el.hasAttribute('disabled') && 
             el.getAttribute('aria-hidden') !== 'true';
    });
  }, []);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;

    const focusableElements = getFocusableElements();
    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      }
    } else {
      if (document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    }
  }, [getFocusableElements]);

  const activate = useCallback((element: HTMLElement | null) => {
    if (!element) return;
    
    containerRef.current = element;
    lastFocusedElement.current = document.activeElement;
    
    const focusableElements = getFocusableElements();
    
    if (focusableElements.length > 0) {
      firstFocusableElement.current = focusableElements[0];
      lastFocusableElement.current = focusableElements[focusableElements.length - 1];
      firstFocusableElement.current?.focus();
    }
    
    document.addEventListener('keydown', handleKeyDown);
  }, [getFocusableElements, handleKeyDown]);

  const deactivate = useCallback(() => {
    document.removeEventListener('keydown', handleKeyDown);
    
    if (lastFocusedElement.current instanceof HTMLElement) {
      (lastFocusedElement.current as HTMLElement).focus();
    }
    
    containerRef.current = null;
    lastFocusedElement.current = null;
    firstFocusableElement.current = null;
    lastFocusableElement.current = null;
  }, [handleKeyDown]);

  useEffect(() => {
    if (!active || !containerRef.current) return;
    
    activate(containerRef.current);
    
    return () => {
      deactivate();
    };
  }, [active, activate, deactivate]);

  return { ref: containerRef, activate, deactivate };
}

// useFocusTrapGroup hook for managing multiple focus traps
export function useFocusTrapGroup({
  active = true,
  onEscape,
}: {
  active?: boolean;
  onEscape?: () => void;
} = {}) {
  const [isActive, setIsActive] = useState(active);
  const containerRef = useRef<HTMLDivElement>(null);
  const { activate, deactivate } = useFocusTrap(isActive);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape' && onEscape) {
        onEscape();
      }
    },
    [onEscape]
  );

  useEffect(() => {
    if (isActive) {
      document.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isActive, handleKeyDown]);

  useEffect(() => {
    if (isActive && containerRef.current) {
      activate(containerRef.current);
    } else {
      deactivate();
    }
  }, [isActive, activate, deactivate]);

  useEffect(() => {
    setIsActive(active);
  }, [active]);

  const activateTrap = useCallback(() => {
    setIsActive(true);
  }, []);

  const deactivateTrap = useCallback(() => {
    setIsActive(false);
  }, []);

  return {
    ref: containerRef,
    isActive,
    activate: activateTrap,
    deactivate: deactivateTrap,
  };
}

// useAutoFocus hook for auto-focusing elements on mount
export function useAutoFocus<T extends HTMLElement>(
  ref: React.RefObject<T>,
  options: { isActive?: boolean; selectText?: boolean } = {}
) {
  const { isActive = true, selectText = false } = options;

  useEffect(() => {
    if (!isActive || !ref.current) return;

    const element = ref.current;
    
    // Focus the element
    element.focus({
      preventScroll: true,
    });

    // Select text if requested and element is an input or textarea
    if (selectText && (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement)) {
      element.select();
    }
  }, [isActive, ref, selectText]);

  return ref;
}
