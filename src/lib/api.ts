import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api'; // Updated to match the Flask backend port

export interface Session {
  id: number;
  title: string;
  type: 'class' | 'event' | 'other';
  time: string;
  location: string;
  instructor: string;
  students: number;
  program: string;
  year: string;
  section: string;
  description: string;
  capacity: string;
  date: string;
}

export const fetchSessions = async (startDate: string, endDate: string): Promise<Session[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/sessions`, {
      params: {
        startDate,
        endDate
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching sessions:', error);
    return [];
  }
};

export const createSession = async (sessionData: Omit<Session, 'id'>): Promise<Session> => {
  try {
    const response = await axios.post(`${API_BASE_URL}/sessions`, sessionData);
    return response.data;
  } catch (error) {
    console.error('Error creating session:', error);
    throw error;
  }
};

export const updateSession = async (id: number, sessionData: Partial<Session>): Promise<Session> => {
  try {
    const response = await axios.put(`${API_BASE_URL}/sessions/${id}`, sessionData);
    return response.data;
  } catch (error) {
    console.error('Error updating session:', error);
    throw error;
  }
};

export const deleteSession = async (id: number): Promise<void> => {
  try {
    await axios.delete(`${API_BASE_URL}/sessions/${id}`);
  } catch (error) {
    console.error('Error deleting session:', error);
    throw error;
  }
};

export interface Student {
  id: number;
  student_id: string;
  firstname: string;
  middle_initial: string;
  surname: string;
  program: string;
  year: string;
  section: string;
  full_name: string;
}

export interface SessionStudentsResponse {
  session: {
    id: number;
    title: string;
    date: string;
    time: string;
    program: string;
    year: string;
    section: string;
  };
  students: Student[];
  count: number;
}

export interface StudentOptions {
  programs: string[];
  years: string[];
  sections: { [key: string]: string[] }; // key is "program|year"
}

export const fetchStudentOptions = async (program?: string, year?: string): Promise<StudentOptions> => {
  try {
    const params = new URLSearchParams();
    if (program) params.append('program', program);
    if (year) params.append('year', year);
    
    const url = `${API_BASE_URL}/students/options${params.toString() ? `?${params.toString()}` : ''}`;
    const response = await axios.get<StudentOptions>(url);
    return response.data;
  } catch (error) {
    console.error('Error fetching student options:', error);
    return { programs: [], years: [], sections: {} };
  }
};

export const fetchSessionStudents = async (sessionId: number): Promise<SessionStudentsResponse> => {
  try {
    const response = await axios.get<SessionStudentsResponse>(`${API_BASE_URL}/sessions/${sessionId}/students`);
    return response.data;
  } catch (error) {
    console.error('Error fetching session students:', error);
    throw error;
  }
};
