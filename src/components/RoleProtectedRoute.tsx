import { Navigate } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import { useState, useEffect, useRef } from "react";
import { fetchUserRole } from "@/lib/getUserRole";

interface RoleProtectedRouteProps {
  children: JSX.Element;
  allowedRoles: string[];
}

// Use the same caching mechanism as navigation component
const getCachedUserRole = (): string | null => {
  try {
    return localStorage.getItem('userRole');
  } catch {
    return null;
  }
};

const getCachedUserId = (): string | null => {
  try {
    return localStorage.getItem('userId');
  } catch {
    return null;
  }
};

const setCachedUserRole = (role: string, userId: string) => {
  try {
    localStorage.setItem('userRole', role);
    localStorage.setItem('userId', userId);
  } catch {
    // Ignore localStorage errors
  }
};

const clearCachedUserRole = () => {
  try {
    localStorage.removeItem('userRole');
    localStorage.removeItem('userId');
  } catch {
    // Ignore localStorage errors
  }
};

let cachedUserRole: string | null = getCachedUserRole();
let cachedUserId: string | null = getCachedUserId();

export const RoleProtectedRoute = ({ children, allowedRoles }: RoleProtectedRouteProps) => {
  const { user, loading: authLoading } = useAuth();
  const [userRole, setUserRole] = useState<string>(() => {
    // Initialize with cached role if available
    return cachedUserRole || 'user';
  });
  const [loading, setLoading] = useState(false); // Start as false since we have cached data
  const isInitialMount = useRef(true);

  useEffect(() => {
    const fetchUserRole = async () => {
      // If we have cached role for the same user, don't refetch
      if (cachedUserRole && cachedUserId === user?.id) {
        setUserRole(cachedUserRole);
        setLoading(false);
        return;
      }

        if (!user) {
          const defaultRole = 'user';
          setUserRole(defaultRole);
          cachedUserRole = defaultRole;
          cachedUserId = null;
          clearCachedUserRole();
          setLoading(false);
          return;
        }

      setLoading(true);
      
      try {
        const role = await fetchUserRole(user.id);
        setUserRole(role);
        
        // Cache the role and user ID both in memory and localStorage
        cachedUserRole = role;
        cachedUserId = user.id;
        setCachedUserRole(role, user.id);
      } catch (error) {
        console.error('Error fetching user role:', error);
        const defaultRole = 'user';
        setUserRole(defaultRole);
        cachedUserRole = defaultRole;
        cachedUserId = user?.id || null;
        if (user?.id) {
          setCachedUserRole(defaultRole, user.id);
        } else {
          clearCachedUserRole();
        }
      } finally {
        setLoading(false);
      }
    };

    // Only fetch role on initial mount or when user changes
    if (!authLoading && (isInitialMount.current || cachedUserId !== user?.id)) {
      fetchUserRole();
      isInitialMount.current = false;
    }
  }, [user?.id, authLoading]);

  if (authLoading || loading) {
    return null; // Return null instead of Loading to prevent flash
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (!userRole || !allowedRoles.includes(userRole)) {
    return <Navigate to="/" replace />;
  }

  return children;
};