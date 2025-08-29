require('@testing-library/jest-dom');
const { TextEncoder, TextDecoder } = require('util');

// Add TextEncoder and TextDecoder to global scope for JSDOM
// These are needed for Supabase client to work in tests
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// Mock the Supabase client
jest.mock('@/lib/supabase', () => {
  const mockAuth = {
    onAuthStateChange: jest.fn(),
    signInWithPassword: jest.fn(),
    signOut: jest.fn(),
    resetPasswordForEmail: jest.fn(),
    updateUser: jest.fn(),
    getSession: jest.fn(),
  };

  const mockStorage = {
    from: jest.fn().mockReturnThis(),
    upload: jest.fn(),
    download: jest.fn(),
    getPublicUrl: jest.fn(),
    remove: jest.fn(),
  };

  const mockFrom = jest.fn().mockReturnThis();
  mockFrom.select = jest.fn().mockReturnThis();
  mockFrom.insert = jest.fn().mockReturnThis();
  mockFrom.update = jest.fn().mockReturnThis();
  mockFrom.delete = jest.fn().mockReturnThis();
  mockFrom.eq = jest.fn().mockReturnThis();
  mockFrom.or_ = jest.fn().mockReturnThis();
  mockFrom.order = jest.fn().mockReturnThis();
  mockFrom.single = jest.fn().mockReturnThis();
  mockFrom.maybeSingle = jest.fn().mockReturnThis();
  mockFrom.execute = jest.fn().mockResolvedValue({ data: [], error: null });

  return {
    __esModule: true,
    default: {
      auth: mockAuth,
      storage: mockStorage,
      from: mockFrom,
    },
    getSupabaseClient: jest.fn().mockReturnThis(),
  };
});

// Mock environment variables
process.env.VITE_SUPABASE_URL = 'https://mock-supabase-url.supabase.co';
process.env.VITE_SUPABASE_ANON_KEY = 'mock-supabase-anon-key';
