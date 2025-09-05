import { supabase } from '../supabase';
import {
  fetchSessions,
  createSession,
  updateSession,
  deleteSession,
  fetchStudents,
  createStudent,
  updateStudent,
  deleteStudent,
  fetchSessionStudents,
  markAttendance,
  getCurrentUser,
  updateUserProfile,
} from '../supabaseService';

// Mock the Supabase client
const mockSupabase = {
  from: jest.fn().mockReturnThis(),
  select: jest.fn().mockReturnThis(),
  insert: jest.fn().mockReturnThis(),
  update: jest.fn().mockReturnThis(),
  delete: jest.fn().mockReturnThis(),
  upsert: jest.fn().mockReturnThis(),
  eq: jest.fn().mockReturnThis(),
  or_: jest.fn().mockReturnThis(),
  order: jest.fn().mockReturnThis(),
  single: jest.fn().mockReturnThis(),
  maybeSingle: jest.fn().mockReturnThis(),
  gte: jest.fn().mockReturnThis(),
  lte: jest.fn().mockReturnThis(),
  storage: {
    from: jest.fn().mockReturnThis(),
    upload: jest.fn().mockResolvedValue({ error: null }),
    getPublicUrl: jest.fn().mockReturnValue({ data: { publicUrl: 'https://example.com/signature.jpg' } }),
  },
  auth: {
    getUser: jest.fn().mockResolvedValue({ data: { user: { id: 'user-123', email: 'test@example.com' } } }),
    updateUser: jest.fn().mockResolvedValue({ data: { user: { id: 'user-123' } }, error: null }),
  },
};

jest.mock('../supabase', () => ({
  __esModule: true,
  supabase: mockSupabase,
}));

describe('supabaseService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('fetchSessions', () => {
    it('should fetch sessions with date range', async () => {
      const mockSessions = [
        { id: 1, title: 'Test Session 1', date: '2023-01-01' },
        { id: 2, title: 'Test Session 2', date: '2023-01-02' },
      ];
      
      (mockSupabase.from('sessions').select as jest.Mock).mockResolvedValueOnce({
        data: mockSessions,
        error: null,
      });

      const result = await fetchSessions('2023-01-01', '2023-01-31');
      
      expect(mockSupabase.from).toHaveBeenCalledWith('sessions');
      expect(mockSupabase.from('sessions').select).toHaveBeenCalledWith('*');
      expect(mockSupabase.from('sessions').gte).toHaveBeenCalledWith('date', '2023-01-01');
      expect(mockSupabase.from('sessions').lte).toHaveBeenCalledWith('date', '2023-01-31');
      expect(result).toEqual(mockSessions);
    });
  });

  describe('createSession', () => {
    it('should create a new session', async () => {
      const newSession = {
        title: 'Test Session',
        type: 'class' as const,
        time_in: '10:00',
        time_out: '11:00',
        time: '10:00 - 11:00',
        location: 'Room 101',
        instructor: 'John Doe',
        program: 'Computer Science',
        year: '2023',
        section: 'A',
        description: 'Test session',
        capacity: '30',
        date: '2023-01-01',
        students: 0,
      };

      const mockResponse = { id: 1, ...newSession };
      (mockSupabase.from('sessions').insert as jest.Mock).mockResolvedValueOnce({
        data: mockResponse,
        error: null,
      });

      const result = await createSession(newSession);
      
      expect(mockSupabase.from('sessions').insert).toHaveBeenCalledWith([newSession]);
      expect(result).toEqual(mockResponse);
    });
  });

  describe('fetchStudents', () => {
    it('should fetch students with filters', async () => {
      const mockStudents = [
        { id: 1, firstname: 'John', surname: 'Doe', program: 'CS', year: '2023', section: 'A' },
        { id: 2, firstname: 'Jane', surname: 'Smith', program: 'CS', year: '2023', section: 'A' },
      ];
      
      (mockSupabase.from('students').select as jest.Mock).mockResolvedValueOnce({
        data: mockStudents,
        error: null,
      });

      const result = await fetchStudents('CS', '2023', 'A');
      
      expect(mockSupabase.from).toHaveBeenCalledWith('students');
      expect(mockSupabase.from('students').select).toHaveBeenCalledWith('*');
      expect(mockSupabase.from('students').eq).toHaveBeenCalledWith('program', 'CS');
      expect(mockSupabase.from('students').eq).toHaveBeenCalledWith('year', '2023');
      expect(mockSupabase.from('students').eq).toHaveBeenCalledWith('section', 'A');
      expect(result).toEqual([
        { ...mockStudents[0], full_name: 'John Doe' },
        { ...mockStudents[1], full_name: 'Jane Smith' },
      ]);
    });
  });

  describe('markAttendance', () => {
    it('should mark attendance for a student', async () => {
      const sessionId = 1;
      const studentId = 1;
      const status = 'present';
      const timeIn = '09:55';
      const timeOut = '11:00';

      const mockResponse = {
        session_id: sessionId,
        student_id: studentId,
        status,
        time_in: timeIn,
        time_out: timeOut,
      };

      (mockSupabase.from('attendance').upsert as jest.Mock).mockResolvedValueOnce({
        data: mockResponse,
        error: null,
      });

      const result = await markAttendance(sessionId, studentId, status, timeIn, timeOut);
      
      expect(mockSupabase.from('attendance').upsert).toHaveBeenCalledWith(
        {
          session_id: sessionId,
          student_id: studentId,
          status,
          time_in: timeIn,
          time_out: timeOut,
          updated_at: expect.any(String),
        },
        { onConflict: 'session_id,student_id' }
      );
      
      expect(result).toEqual(mockResponse);
    });
  });

  describe('getCurrentUser', () => {
    it('should return the current user', async () => {
      const mockUser = {
        id: 'user-123',
        email: 'test@example.com',
        user_metadata: { full_name: 'Test User' },
      };
      
      const mockProfile = {
        id: 'user-123',
        full_name: 'Test User',
        role: 'admin',
        avatar_url: 'https://example.com/avatar.jpg',
      };

      (mockSupabase.auth.getUser as jest.Mock).mockResolvedValueOnce({
        data: { user: mockUser },
      });

      (mockSupabase.from('profiles').select as jest.Mock).mockResolvedValueOnce({
        data: mockProfile,
        error: null,
      });

      const result = await getCurrentUser();
      
      expect(result).toEqual({
        id: 'user-123',
        email: 'test@example.com',
        name: 'Test User',
        role: 'admin',
        avatar_url: 'https://example.com/avatar.jpg',
      });
    });
  });

  describe('fetchSessionStudents', () => {
    it('should fetch students for a session with specific filters', async () => {
      const mockSession = {
        id: 1,
        title: 'Test Session',
        program: 'Computer Science',
        year: '1st',
        section: 'A',
        date: '2023-01-01',
        time_in: '10:00',
        time_out: '11:00',
        description: 'Test session',
        type: 'class'
      };

      const mockStudents = [
        { id: 1, firstname: 'John', surname: 'Doe', program: 'Computer Science', year: '1st', section: 'A' },
        { id: 2, firstname: 'Jane', surname: 'Smith', program: 'Computer Science', year: '1st', section: 'A' }
      ];

      // Mock session fetch
      (mockSupabase.from('sessions').select as jest.Mock).mockResolvedValueOnce({
        data: mockSession,
        error: null,
      });

      // Mock student count
      (mockSupabase.from('students').select as jest.Mock).mockResolvedValueOnce({
        count: 2,
        error: null,
      });

      // Mock student fetch
      (mockSupabase.from('students').select as jest.Mock).mockResolvedValueOnce({
        data: mockStudents,
        error: null,
      });

      // Mock attendance fetch
      (mockSupabase.from('attendance').select as jest.Mock).mockResolvedValueOnce({
        data: [],
        error: null,
      });

      const result = await fetchSessionStudents(1);

      expect(result.session.id).toBe(1);
      expect(result.students).toHaveLength(2);
      expect(result.count).toBe(2);
    });

    it('should handle "All Programs" selection correctly', async () => {
      const mockSession = {
        id: 1,
        title: 'Test Session',
        program: 'All Programs',
        year: '1st',
        section: 'A',
        date: '2023-01-01',
        time_in: '10:00',
        time_out: '11:00',
        description: 'Test session',
        type: 'class'
      };

      const mockStudents = [
        { id: 1, firstname: 'John', surname: 'Doe', program: 'Computer Science', year: '1st', section: 'A' },
        { id: 2, firstname: 'Jane', surname: 'Smith', program: 'Engineering', year: '1st', section: 'A' }
      ];

      // Mock session fetch
      (mockSupabase.from('sessions').select as jest.Mock).mockResolvedValueOnce({
        data: mockSession,
        error: null,
      });

      // Mock student count (should not filter by program)
      (mockSupabase.from('students').select as jest.Mock).mockResolvedValueOnce({
        count: 2,
        error: null,
      });

      // Mock student fetch (should not filter by program)
      (mockSupabase.from('students').select as jest.Mock).mockResolvedValueOnce({
        data: mockStudents,
        error: null,
      });

      // Mock attendance fetch
      (mockSupabase.from('attendance').select as jest.Mock).mockResolvedValueOnce({
        data: [],
        error: null,
      });

      const result = await fetchSessionStudents(1);

      expect(result.session.id).toBe(1);
      expect(result.students).toHaveLength(2);
      expect(result.count).toBe(2);
    });

    it('should handle "All Year Levels" selection correctly', async () => {
      const mockSession = {
        id: 1,
        title: 'Test Session',
        program: 'Computer Science',
        year: 'All Year Levels',
        section: 'A',
        date: '2023-01-01',
        time_in: '10:00',
        time_out: '11:00',
        description: 'Test session',
        type: 'class'
      };

      const mockStudents = [
        { id: 1, firstname: 'John', surname: 'Doe', program: 'Computer Science', year: '1st', section: 'A' },
        { id: 2, firstname: 'Jane', surname: 'Smith', program: 'Computer Science', year: '2nd', section: 'A' }
      ];

      // Mock session fetch
      (mockSupabase.from('sessions').select as jest.Mock).mockResolvedValueOnce({
        data: mockSession,
        error: null,
      });

      // Mock student count (should not filter by year)
      (mockSupabase.from('students').select as jest.Mock).mockResolvedValueOnce({
        count: 2,
        error: null,
      });

      // Mock student fetch (should not filter by year)
      (mockSupabase.from('students').select as jest.Mock).mockResolvedValueOnce({
        data: mockStudents,
        error: null,
      });

      // Mock attendance fetch
      (mockSupabase.from('attendance').select as jest.Mock).mockResolvedValueOnce({
        data: [],
        error: null,
      });

      const result = await fetchSessionStudents(1);

      expect(result.session.id).toBe(1);
      expect(result.students).toHaveLength(2);
      expect(result.count).toBe(2);
    });

    it('should handle "All Sections" selection correctly', async () => {
      const mockSession = {
        id: 1,
        title: 'Test Session',
        program: 'Computer Science',
        year: '1st',
        section: 'All Sections',
        date: '2023-01-01',
        time_in: '10:00',
        time_out: '11:00',
        description: 'Test session',
        type: 'class'
      };

      const mockStudents = [
        { id: 1, firstname: 'John', surname: 'Doe', program: 'Computer Science', year: '1st', section: 'A' },
        { id: 2, firstname: 'Jane', surname: 'Smith', program: 'Computer Science', year: '1st', section: 'B' }
      ];

      // Mock session fetch
      (mockSupabase.from('sessions').select as jest.Mock).mockResolvedValueOnce({
        data: mockSession,
        error: null,
      });

      // Mock student count (should not filter by section)
      (mockSupabase.from('students').select as jest.Mock).mockResolvedValueOnce({
        count: 2,
        error: null,
      });

      // Mock student fetch (should not filter by section)
      (mockSupabase.from('students').select as jest.Mock).mockResolvedValueOnce({
        data: mockStudents,
        error: null,
      });

      // Mock attendance fetch
      (mockSupabase.from('attendance').select as jest.Mock).mockResolvedValueOnce({
        data: [],
        error: null,
      });

      const result = await fetchSessionStudents(1);

      expect(result.session.id).toBe(1);
      expect(result.students).toHaveLength(2);
      expect(result.count).toBe(2);
    });
  });
});