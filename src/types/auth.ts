import type { User, Session } from '@supabase/supabase-js';

export type AuthError = {
  message: string;
  status?: number;
  code?: string;
  name?: string;
};

export type AuthResponse = {
  error: AuthError | null;
  data?: {
    user?: User;
    session?: Session;
  };
};

export type AuthContextType = {
  // State
  user: User | null;
  session: Session | null;
  loading: boolean;
  error: string | null;
  
  // Methods
  signIn: (email: string, password: string) => Promise<AuthResponse>;
  signUp: (email: string, password: string, firstName: string, lastName: string, role: string) => Promise<AuthResponse>;
  signOut: () => Promise<{ error: AuthError | null }>;
  resetPassword: (email: string) => Promise<{ error: AuthError | null }>;
  updatePassword: (newPassword: string) => Promise<{ error: AuthError | null }>;
  
  // Helpers
  isAuthenticated: boolean;
  hasRole: (role: string) => boolean;
};
