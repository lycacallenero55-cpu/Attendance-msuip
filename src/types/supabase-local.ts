// Local type definitions to work around Supabase type generation issues
// This is a temporary fix while Supabase types are being regenerated

export interface LocalSession {
  id: number;
  title: string;
  type: 'class' | 'event' | 'other';
  time_in?: string;
  time_out?: string;
  created_by_user_id?: string;
  program: string;
  year: string;
  section: string;
  description?: string;
  capacity: string | number;
  date: string;
  created_at: string;
  updated_at: string;
}

export interface LocalStudent {
  id: number;
  student_id: string;
  firstname: string;
  surname: string;
  middlename?: string;
  middle_initial?: string;
  program: string;
  year: string;
  section: string;
  signature_url?: string;
  signature_urls?: string[];
  signature_count?: number;
  primary_signature_id?: number;
  email?: string;
  contact_no?: string;
  address?: string;
  sex?: string;
  birthday?: string;
  created_at: string;
  updated_at: string;
  full_name?: string;
}

export interface LocalProfile {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  role: 'admin' | 'staff' | 'ssg_officer';
  status: 'pending' | 'active' | 'inactive';
  position?: string;
  department?: string;
  created_at: string;
  updated_at: string;
}

export interface LocalAttendance {
  id: number;
  session_id: number;
  student_id: number;
  status: 'present' | 'absent' | 'late' | 'excused';
  time_in?: string;
  time_out?: string;
  created_at: string;
  updated_at: string;
}

export interface LocalSignature {
  id: number;
  student_id: number;
  storage_path: string;
  file_name: string;
  file_size: number;
  file_type: string;
  width?: number;
  height?: number;
  features?: Record<string, unknown>;
  quality_score?: number;
  device_info?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}