import * as React from 'react';
import { useCallback, useEffect, useRef, useState, ReactNode } from 'react';

// Helper function to handle class names with proper type safety
type ClassValue = string | number | boolean | null | undefined;

function cn(...classes: ClassValue[]): string {
  return classes
    .filter((c): c is string | number | boolean => c !== null && c !== undefined)
    .map(c => c.toString().trim())
    .filter(Boolean)
    .join(' ');
}

/**
 * VisuallyHidden component for screen reader only content
 */
export const VisuallyHidden = ({
  children,
  className = '',
  ...delegated
}: {
  children: React.ReactNode;
  className?: string;
  [key: string]: unknown;
}) => {
  const [forceShow, setForceShow] = useState(false);
  const isDev = process.env.NODE_ENV === 'development';

  useEffect(() => {
    if (!isDev) return;

    const keyDownHandler = (e: KeyboardEvent) => {
      if (e.key === 'Alt') {
        setForceShow(true);
      }
    };

    const keyUpHandler = (e: KeyboardEvent) => {
      if (e.key === 'Alt') {
        setForceShow(false);
      }
    };

    // Use the same function reference for both add and remove
    window.addEventListener('keydown', keyDownHandler);
    window.addEventListener('keyup', keyUpHandler);

    return () => {
      window.removeEventListener('keydown', keyDownHandler);
      window.removeEventListener('keyup', keyUpHandler);
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
      className={cn(
        'absolute -m-px h-px w-px overflow-hidden border-0 p-0',
        'whitespace-nowrap',
        'clip-[rect(0,0,0,0)]',
        className
      )}
      {...delegated}
    >
      {children}
    </span>
  );
};

/**
 * SkipLink component for keyboard navigation
 */
export const SkipLink = ({
  targetId = 'main-content',
  children = 'Skip to content',
  className = '',
  ...delegated
}: {
  targetId?: string;
  children?: React.ReactNode;
  className?: string;
  [key: string]: unknown;
}) => {
  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    const target = document.getElementById(targetId);
    
    if (target) {
      target.setAttribute('tabindex', '-1');
      target.focus();
      
      // Remove the tabindex after focus is set
      setTimeout(() => {
        target.removeAttribute('tabindex');
      }, 0);
    }
  };

  return (
    <a
      href={`#${targetId}`}
      onClick={handleClick}
      className={cn(
        'fixed left-0 top-0 z-50 -translate-y-full transform bg-primary px-4 py-2 text-sm font-medium text-white',
        'transition-transform duration-200 focus:translate-y-0',
        className
      )}
      {...delegated}
    >
      {children}
    </a>
  );
};

/**
 * FocusTrap component to trap focus within a specific area
 */
export const FocusTrap = ({
  children,
  active = true,
  className = '',
  ...delegated
}: {
  children: React.ReactNode;
  active?: boolean;
  className?: string;
  [key: string]: unknown;
}) => {
  const trapRef = useRef<HTMLDivElement>(null);
  const focusableElements = useRef<HTMLElement[]>([]);
  const firstFocusableElement = useRef<HTMLElement | null>(null);
  const lastFocusableElement = useRef<HTMLElement | null>(null);

  const getFocusableElements = useCallback(() => {
    if (!trapRef.current) return [];
    
    return Array.from(
      trapRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )
    ).filter(el => !el.hasAttribute('disabled') && !el.getAttribute('aria-hidden'));
  }, []);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!active || !trapRef.current || !focusableElements.current.length) return;
    
    const isTabPressed = e.key === 'Tab' || e.keyCode === 9;
    if (!isTabPressed) return;

    if (e.shiftKey) {
      if (document.activeElement === firstFocusableElement.current) {
        e.preventDefault();
        lastFocusableElement.current?.focus();
      }
    } else {
      if (document.activeElement === lastFocusableElement.current) {
        e.preventDefault();
        firstFocusableElement.current?.focus();
      }
    }
  }, [active]);

  useEffect(() => {
    if (!active) return;
    
    // Get all focusable elements
    focusableElements.current = getFocusableElements();
    
    if (focusableElements.current.length > 0) {
      firstFocusableElement.current = focusableElements.current[0];
      lastFocusableElement.current = focusableElements.current[focusableElements.current.length - 1];
      
      // Focus the first element
      firstFocusableElement.current.focus();
    }
    
    // Add event listener for keyboard navigation
    const keyDownHandler = handleKeyDown as EventListener;
    document.addEventListener('keydown', keyDownHandler);
    
    return () => {
      document.removeEventListener('keydown', keyDownHandler);
    };
  }, [active, getFocusableElements, handleKeyDown]);

  return (
    <div ref={trapRef} className={className} {...delegated}>
      {children}
    </div>
  );
};

/**
 * useFocusTrap hook for more control over focus trapping
 */
export const useFocusTrap = (active = true) => {
  const containerRef = useRef<HTMLElement>(null);
  const [focusableElements, setFocusableElements] = useState<HTMLElement[]>([]);
  const firstFocusableElement = useRef<HTMLElement | null>(null);
  const lastFocusableElement = useRef<HTMLElement | null>(null);

  const getFocusableElements = useCallback(() => {
    if (!containerRef.current) return [];
    
    return Array.from(
      containerRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )
    ).filter(el => !el.hasAttribute('disabled') && !el.getAttribute('aria-hidden'));
  }, []);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!active || !containerRef.current || !focusableElements.length) return;
    
    const isTabPressed = e.key === 'Tab' || e.keyCode === 9;
    if (!isTabPressed) return;

    if (e.shiftKey) {
      if (document.activeElement === firstFocusableElement.current) {
        e.preventDefault();
        lastFocusableElement.current?.focus();
      }
    } else {
      if (document.activeElement === lastFocusableElement.current) {
        e.preventDefault();
        firstFocusableElement.current?.focus();
      }
    }
  }, [active, focusableElements]);

  useEffect(() => {
    if (!active) return;
    
    // Get all focusable elements
    const elements = getFocusableElements();
    setFocusableElements(elements);
    
    if (elements.length > 0) {
      firstFocusableElement.current = elements[0];
      lastFocusableElement.current = elements[elements.length - 1];
      
      // Focus the first element
      firstFocusableElement.current.focus();
    }
    
    // Add event listener for keyboard navigation
    const keyDownHandler = handleKeyDown as EventListener;
    document.addEventListener('keydown', keyDownHandler);
    
    return () => {
      document.removeEventListener('keydown', keyDownHandler);
    };
  }, [active, getFocusableElements, handleKeyDown]);

  return containerRef;
};

/**
 * useFocusOnMount hook to focus an element when it mounts
 */
export const useFocusOnMount = <T extends HTMLElement = HTMLElement>(
  shouldFocus = true,
  options: FocusOptions = { preventScroll: true }
) => {
  const ref = useRef<T>(null);

  useEffect(() => {
    if (shouldFocus && ref.current) {
      ref.current.focus(options);
    }
  }, [shouldFocus, options]);

  return ref;
};

/**
 * useAutoFocus hook to automatically focus an element when it mounts
 */
export const useAutoFocus = <T extends HTMLElement = HTMLElement>(
  autoFocus = true,
  options: FocusOptions = { preventScroll: true }
) => {
  return useFocusOnMount<T>(autoFocus, options);
};

/**
 * useFocusTrapGroup hook for managing focus within a group of focusable elements
 */
export const useFocusTrapGroup = ({
  initialFocusRef,
  onEscape,
  returnFocusOnDeactivate = true,
  preventScrollOnDeactivate = false,
}: {
  initialFocusRef?: React.RefObject<HTMLElement>;
  onEscape?: () => void;
  returnFocusOnDeactivate?: boolean;
  preventScrollOnDeactivate?: boolean;
} = {}) => {
  const [isActive, setIsActive] = useState(false);
  const lastFocusedElement = useRef<Element | null>(null);
  const containerRef = useRef<HTMLElement>(null);
  const focusTrapRef = useFocusTrap(isActive);

  // Handle escape key
  useEffect(() => {
    if (!isActive || !onEscape) return;

    const handleKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') {
        onEscape();
      }
    };

    const keyDownHandler = handleKeyDown as EventListener;
    document.addEventListener('keydown', keyDownHandler);
    return () => {
      document.removeEventListener('keydown', keyDownHandler);
    };
  }, [isActive, onEscape]);

  // Activate the focus trap
  const activate = useCallback(() => {
    if (isActive) return;
    
    // Save the currently focused element
    lastFocusedElement.current = document.activeElement;
    
    // Set active
    setIsActive(true);
    
    // Focus the initial element if provided
    if (initialFocusRef?.current) {
      initialFocusRef.current.focus();
    }
  }, [initialFocusRef, isActive]);

  // Deactivate the focus trap
  const deactivate = useCallback(() => {
    if (!isActive) return;
    
    // Return focus to the last focused element
    if (returnFocusOnDeactivate && lastFocusedElement.current instanceof HTMLElement) {
      lastFocusedElement.current.focus({ preventScroll: preventScrollOnDeactivate });
    }
    
    // Reset
    lastFocusedElement.current = null;
    setIsActive(false);
  }, [isActive, preventScrollOnDeactivate, returnFocusOnDeactivate]);

  // Set up ref forwarding
  const setRefs = useCallback((node: HTMLElement | null) => {
    // Set the container ref
    containerRef.current = node;
    
    // Forward the ref to the focus trap
    if (focusTrapRef && typeof focusTrapRef === 'object') {
      (focusTrapRef as React.MutableRefObject<HTMLElement | null>).current = node;
    }
  }, [focusTrapRef]);

  return {
    ref: setRefs,
    active: isActive,
    activate,
    deactivate,
    toggle: isActive ? deactivate : activate,
  };
};

export default {
  VisuallyHidden,
  SkipLink,
  FocusTrap,
  useFocusTrap,
  useFocusOnMount,
  useAutoFocus,
  useFocusTrapGroup,
};
