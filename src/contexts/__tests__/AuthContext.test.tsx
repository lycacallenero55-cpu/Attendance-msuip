import { render, screen, waitFor, act } from '@testing-library/react';
import { AuthProvider } from '../AuthContext';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/lib/supabase';
import type { AuthResponse } from '@/types/auth';

// Mock the supabase auth methods
const mockSignIn = jest.fn();
const mockSignOut = jest.fn();
const mockResetPassword = jest.fn();
const mockUpdateUser = jest.fn();
const mockGetSession = jest.fn();

// Mock the supabase client
jest.mock('@/lib/supabase', () => ({
  __esModule: true,
  supabase: {
    auth: {
      signInWithPassword: jest.fn().mockImplementation(mockSignIn),
      signOut: jest.fn().mockImplementation(mockSignOut),
      resetPasswordForEmail: jest.fn().mockImplementation(mockResetPassword),
      updateUser: jest.fn().mockImplementation(mockUpdateUser),
      onAuthStateChange: jest.fn((callback) => {
        // Simulate auth state change
        callback('SIGNED_IN', { user: { id: '123', email: 'test@example.com' } });
        return { data: { subscription: { unsubscribe: jest.fn() } } };
      }),
      getSession: jest.fn().mockImplementation(mockGetSession),
    },
  },
}));

describe('AuthProvider', () => {
  const TestComponent = () => {
    const { user, isAuthenticated, loading } = useAuth();
    return (
      <div>
        <div data-testid="user-email">{user?.email}</div>
        <div data-testid="is-authenticated">{isAuthenticated ? 'true' : 'false'}</div>
        <div data-testid="is-loading">{loading ? 'true' : 'false'}</div>
      </div>
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockGetSession.mockResolvedValue({ data: { session: null }, error: null });
  });

  it('should provide initial auth state', async () => {
    render(<TestComponent />);
    
    // Initial loading state should be true
    expect(screen.getByTestId('is-loading').textContent).toBe('true');
    
    // After auth state is determined, loading should be false
    await waitFor(() => {
      expect(screen.getByTestId('is-loading').textContent).toBe('false');
    });
    
    // Should be authenticated after auth state change
    expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
    expect(screen.getByTestId('user-email').textContent).toBe('test@example.com');
  });

  it('should handle sign in', async () => {
    // Mock successful sign in
    mockSignIn.mockResolvedValueOnce({
      data: {
        user: { id: '123', email: 'test@example.com' },
        session: { access_token: 'test-token' },
      },
      error: null,
    });

    let signInFunction: (email: string, password: string) => Promise<AuthResponse>;
    
    const TestSignInComponent = () => {
      const { signIn } = useAuth();
      signInFunction = signIn;
      return null;
    };
    
    render(
      <AuthProvider>
        <TestSignInComponent />
      </AuthProvider>
    );

    await act(async () => {
      await signInFunction('test@example.com', 'password');
    });

    expect(mockSignIn).toHaveBeenCalledWith({
      email: 'test@example.com',
      password: 'password',
    });
  });

  it('should handle sign out', async () => {
    // Mock successful sign out
    mockSignOut.mockResolvedValueOnce({ error: null });

    let signOutFunction: () => Promise<{ error: { message: string } | null }>;
    
    const TestSignOutComponent = () => {
      const { signOut } = useAuth();
      signOutFunction = signOut;
      return null;
    };
    
    render(
      <AuthProvider>
        <TestSignOutComponent />
      </AuthProvider>
    );

    await act(async () => {
      await signOutFunction();
    });

    expect(mockSignOut).toHaveBeenCalled();
  });

  it('should handle password reset', async () => {
    // Mock successful password reset
    mockResetPassword.mockResolvedValueOnce({ error: null });

    let resetPasswordFunction: (email: string) => Promise<{ error: { message: string } | null }>;
    
    const TestResetPasswordComponent = () => {
      const { resetPassword } = useAuth();
      resetPasswordFunction = resetPassword;
      return null;
    };
    
    render(
      <AuthProvider>
        <TestResetPasswordComponent />
      </AuthProvider>
    );

    await act(async () => {
      await resetPasswordFunction('test@example.com');
    });

    expect(mockResetPassword).toHaveBeenCalledWith('test@example.com', {
      redirectTo: expect.stringContaining('/update-password'),
    });
  });

  it('should handle password update', async () => {
    // Mock successful password update
    mockUpdateUser.mockResolvedValueOnce({ data: { user: {} }, error: null });

    let updatePasswordFunction: (password: string) => Promise<{ error: { message: string } | null }>;
    
    // Mock the current user
    mockGetSession.mockResolvedValueOnce({
      data: { session: { user: { id: '123', email: 'test@example.com' } } },
      error: null,
    });

    const TestUpdatePasswordComponent = () => {
      const { updatePassword } = useAuth();
      updatePasswordFunction = updatePassword;
      return null;
    };
    
    render(
      <AuthProvider>
        <TestUpdatePasswordComponent />
      </AuthProvider>
    );

    await act(async () => {
      await updatePasswordFunction('new-password');
    });

    expect(mockUpdateUser).toHaveBeenCalledWith({
      password: 'new-password',
    });
  });

  it('should handle auth errors', async () => {
    // Mock error response
    const testError = new Error('Test error');
    mockSignIn.mockRejectedValueOnce(testError);

    let signInFunction: (email: string, password: string) => Promise<AuthResponse>;
    let errorState: string | null;
    
    const TestErrorComponent = () => {
      const { signIn, error } = useAuth();
      signInFunction = signIn;
      errorState = error;
      return null;
    };
    
    render(
      <AuthProvider>
        <TestErrorComponent />
      </AuthProvider>
    );

    await act(async () => {
      await signInFunction('test@example.com', 'wrong-password');
    });

    expect(errorState).toBe('Test error');
  });
});
