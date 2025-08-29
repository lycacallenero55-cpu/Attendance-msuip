-- Create missing tables for the attendance system

-- 1. Create excuse_applications table
CREATE TABLE public.excuse_applications (
  id BIGSERIAL PRIMARY KEY,
  student_id BIGINT NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  session_id BIGINT REFERENCES public.sessions(id) ON DELETE SET NULL,
  absence_date DATE NOT NULL,
  reason TEXT NOT NULL,
  documentation_url TEXT,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
  reviewed_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  reviewed_at TIMESTAMP WITH TIME ZONE,
  review_notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- 2. Create academic_years table
CREATE TABLE public.academic_years (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL UNIQUE,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT false,
  description TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  CONSTRAINT valid_academic_year_dates CHECK (end_date > start_date)
);

-- 3. Create semesters table
CREATE TABLE public.semesters (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  academic_year_id UUID NOT NULL REFERENCES public.academic_years(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT false,
  description TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  CONSTRAINT valid_semester_dates CHECK (end_date > start_date)
);

-- 4. Create attendance_records table for detailed records management
CREATE TABLE public.attendance_records (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id BIGINT NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  session_id BIGINT NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
  attendance_id BIGINT REFERENCES public.attendance(id) ON DELETE SET NULL,
  date DATE NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('present', 'absent', 'late', 'excused')),
  time_in TIMESTAMP WITH TIME ZONE,
  time_out TIMESTAMP WITH TIME ZONE,
  notes TEXT,
  recorded_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  UNIQUE(student_id, session_id, date)
);

-- 5. Add missing columns to existing tables
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS department TEXT;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS position TEXT;

-- Enable RLS on all tables
ALTER TABLE public.excuse_applications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.academic_years ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.semesters ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.attendance_records ENABLE ROW LEVEL SECURITY;

-- Create RLS policies

-- excuse_applications policies
CREATE POLICY "Users can view excuse applications for their students" 
ON public.excuse_applications 
FOR SELECT 
USING (
  EXISTS (
    SELECT 1 FROM public.profiles 
    WHERE id = auth.uid() AND role IN ('admin', 'instructor')
  )
);

CREATE POLICY "Instructors can create excuse applications" 
ON public.excuse_applications 
FOR INSERT 
WITH CHECK (
  EXISTS (
    SELECT 1 FROM public.profiles 
    WHERE id = auth.uid() AND role IN ('admin', 'instructor')
  )
);

CREATE POLICY "Instructors can update excuse applications" 
ON public.excuse_applications 
FOR UPDATE 
USING (
  EXISTS (
    SELECT 1 FROM public.profiles 
    WHERE id = auth.uid() AND role IN ('admin', 'instructor')
  )
);

-- academic_years policies
CREATE POLICY "Users can view academic years" 
ON public.academic_years 
FOR SELECT 
USING (auth.role() = 'authenticated');

CREATE POLICY "Admins can manage academic years" 
ON public.academic_years 
FOR ALL 
USING (
  EXISTS (
    SELECT 1 FROM public.profiles 
    WHERE id = auth.uid() AND role = 'admin'
  )
);

-- semesters policies
CREATE POLICY "Users can view semesters" 
ON public.semesters 
FOR SELECT 
USING (auth.role() = 'authenticated');

CREATE POLICY "Admins can manage semesters" 
ON public.semesters 
FOR ALL 
USING (
  EXISTS (
    SELECT 1 FROM public.profiles 
    WHERE id = auth.uid() AND role = 'admin'
  )
);

-- attendance_records policies
CREATE POLICY "Users can view attendance records" 
ON public.attendance_records 
FOR SELECT 
USING (auth.role() = 'authenticated');

CREATE POLICY "Instructors can manage attendance records" 
ON public.attendance_records 
FOR ALL 
USING (
  EXISTS (
    SELECT 1 FROM public.profiles 
    WHERE id = auth.uid() AND role IN ('admin', 'instructor')
  )
);

-- Create indexes for better performance
CREATE INDEX idx_excuse_applications_student_id ON public.excuse_applications(student_id);
CREATE INDEX idx_excuse_applications_session_id ON public.excuse_applications(session_id);
CREATE INDEX idx_excuse_applications_status ON public.excuse_applications(status);
CREATE INDEX idx_excuse_applications_absence_date ON public.excuse_applications(absence_date);

CREATE INDEX idx_academic_years_is_active ON public.academic_years(is_active);
CREATE INDEX idx_academic_years_start_date ON public.academic_years(start_date);

CREATE INDEX idx_semesters_academic_year_id ON public.semesters(academic_year_id);
CREATE INDEX idx_semesters_is_active ON public.semesters(is_active);

CREATE INDEX idx_attendance_records_student_id ON public.attendance_records(student_id);
CREATE INDEX idx_attendance_records_session_id ON public.attendance_records(session_id);
CREATE INDEX idx_attendance_records_date ON public.attendance_records(date);

-- Create triggers for updated_at columns
CREATE TRIGGER update_excuse_applications_updated_at
  BEFORE UPDATE ON public.excuse_applications
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_academic_years_updated_at
  BEFORE UPDATE ON public.academic_years
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_semesters_updated_at
  BEFORE UPDATE ON public.semesters
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_attendance_records_updated_at
  BEFORE UPDATE ON public.attendance_records
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

-- Insert default academic year data
INSERT INTO public.academic_years (name, start_date, end_date, is_active, description)
VALUES 
  ('2024-2025', '2024-06-01', '2025-05-31', true, 'Current Academic Year'),
  ('2023-2024', '2023-06-01', '2024-05-31', false, 'Previous Academic Year');

-- Insert default semester data
INSERT INTO public.semesters (academic_year_id, name, start_date, end_date, is_active, description)
VALUES 
  ((SELECT id FROM public.academic_years WHERE name = '2024-2025'), '1st Semester', '2024-06-01', '2024-10-31', true, 'First Semester 2024-2025'),
  ((SELECT id FROM public.academic_years WHERE name = '2024-2025'), '2nd Semester', '2024-11-01', '2025-03-31', false, 'Second Semester 2024-2025'),
  ((SELECT id FROM public.academic_years WHERE name = '2024-2025'), 'Summer', '2025-04-01', '2025-05-31', false, 'Summer 2025');