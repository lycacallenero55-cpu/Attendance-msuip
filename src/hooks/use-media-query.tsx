import { useState, useEffect } from 'react';

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState<boolean>(false);

  useEffect(() => {
    // Check if window is defined (prevents SSR issues)
    if (typeof window === 'undefined') return;

    const mediaQuery = window.matchMedia(query);
    
    // Update the state with the current value
    setMatches(mediaQuery.matches);
    
    // Create a callback function to handle changes
    const listener = (event: MediaQueryListEvent) => {
      setMatches(event.matches);
    };
    
    // Add the listener for changes
    mediaQuery.addEventListener('change', listener);
    
    // Clean up the listener when the component unmounts
    return () => {
      mediaQuery.removeEventListener('change', listener);
    };
  }, [query]);

  return matches;
}
