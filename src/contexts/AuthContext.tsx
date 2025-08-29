import { useEffect, useState, useCallback, useContext } from 'react';
import { User, Session, AuthError } from '@supabase/supabase-js';
import { supabase } from '@/lib/supabase';
import type { AuthContextType } from '@/types/auth';
import { AuthContext } from './auth-context';

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Handle auth state changes
  useEffect(() => {
    // Get initial session
    const getInitialSession = async () => {
      try {
        const { data: { session: initialSession } } = await supabase.auth.getSession();
        setSession(initialSession);
        setUser(initialSession?.user ?? null);
      } catch (err) {
        setError('Failed to get initial session');
        console.error('Error getting initial session:', err);
      } finally {
        setLoading(false);
      }
    };

    getInitialSession();

    // Listen for auth state changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, newSession) => {
        setSession(newSession);
        setUser(newSession?.user ?? null);
        setError(null);
        setLoading(false);
      }
    );

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  // Sign in with email and password
  const signIn = useCallback(async (email: string, password: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const { data, error: signInError } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      
      if (signInError) {
        setError(signInError.message);
        return { error: signInError };
      }
      
      setUser(data.user);
      setSession(data.session);
      return { data, error: null };
    } catch (err) {
      const error = err as AuthError;
      setError(error.message || 'Failed to sign in');
      return { error };
    } finally {
      setLoading(false);
    }
  }, []);

  // Sign up with email and password
  const signUp = useCallback(async (email: string, password: string, firstName: string, lastName: string, role: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const { data, error: signUpError } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            first_name: firstName,
            last_name: lastName,
            role: role,
          },
          emailRedirectTo: `${window.location.origin}/`,
        },
      });
      
      if (signUpError) {
        setError(signUpError.message);
        return { error: signUpError };
      }
      // Insert into users table (non-admin accounts only)
      if (data?.user) {
        const normalizedRole = role; // role already chosen from allowed set
        const { error: insertErr } = await supabase
          .from('users')
          .insert([{
            id: data.user.id,
            email,
            first_name: firstName,
            last_name: lastName,
            role: normalizedRole,
          }]);
        if (insertErr) {
          console.error('Error inserting users row:', insertErr);
        }
      }

      return { data, error: null };
    } catch (err) {
      const error = err as AuthError;
      setError(error.message || 'Failed to sign up');
      return { error };
    } finally {
      setLoading(false);
    }
  }, []);

  // Sign out
  const signOut = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const { error: signOutError } = await supabase.auth.signOut();
      
      if (signOutError) {
        setError(signOutError.message);
        return { error: signOutError };
      }
      
      setUser(null);
      setSession(null);
      return { error: null };
    } catch (err) {
      const error = err as AuthError;
      setError(error.message || 'Failed to sign out');
      return { error };
    } finally {
      setLoading(false);
    }
  }, []);

  // Reset password
  const resetPassword = useCallback(async (email: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const { error: resetError } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/update-password`,
      });
      
      if (resetError) {
        setError(resetError.message);
        return { error: resetError };
      }
      
      return { error: null };
    } catch (err) {
      const error = err as AuthError;
      setError(error.message || 'Failed to reset password');
      return { error };
    } finally {
      setLoading(false);
    }
  }, []);

  // Update password
  const updatePassword = useCallback(async (newPassword: string) => {
    if (!user) {
      const error = new Error('User not authenticated') as AuthError;
      setError(error.message);
      return { error };
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const { error: updateError } = await supabase.auth.updateUser({
        password: newPassword,
      });
      
      if (updateError) {
        setError(updateError.message);
        return { error: updateError };
      }
      
      return { error: null };
    } catch (err) {
      const error = err as AuthError;
      setError(error.message || 'Failed to update password');
      return { error };
    } finally {
      setLoading(false);
    }
  }, [user]);

  const isAuthenticated = !!user;
  
  const hasRole = useCallback((role: string): boolean => {
    if (!user) return false;
    // Check user's app_metadata or user_metadata for roles
    return user.app_metadata?.roles?.includes(role) || false;
  }, [user]);

  const value = {
    // State
    user,
    session,
    loading,
    error,
    
    // Methods
    signIn,
    signUp,
    signOut,
    resetPassword,
    updatePassword,
    
    // Helpers
    isAuthenticated,
    hasRole,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
