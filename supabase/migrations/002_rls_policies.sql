-- Enable Row Level Security on all tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.students ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.attendance ENABLE ROW LEVEL SECURITY;

-- Create a function to check if current user is an admin
CREATE OR REPLACE FUNCTION is_admin()
RETURNS BOOLEAN AS $$
  SELECT EXISTS (
    SELECT 1 FROM profiles 
    WHERE id = auth.uid() AND role = 'admin'
  );
$$ LANGUAGE sql SECURITY DEFINER;

-- Create a function to check if current user is an instructor
CREATE OR REPLACE FUNCTION is_instructor()
RETURNS BOOLEAN AS $$
  SELECT EXISTS (
    SELECT 1 FROM profiles 
    WHERE id = auth.uid() AND role IN ('admin', 'instructor')
  );
$$ LANGUAGE sql SECURITY DEFINER;

-- Profiles policies
CREATE POLICY "Users can view their own profile"
  ON profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
  ON profiles FOR UPDATE
  USING (auth.uid() = id);

-- Sessions policies
CREATE POLICY "Enable read access for all users"
  ON sessions FOR SELECT
  USING (true);

CREATE POLICY "Enable insert for instructors and admins"
  ON sessions FOR INSERT
  WITH CHECK (is_instructor());

CREATE POLICY "Enable update for instructors and admins"
  ON sessions FOR UPDATE
  USING (is_instructor());

CREATE POLICY "Enable delete for admins only"
  ON sessions FOR DELETE
  USING (is_admin());

-- Students policies
CREATE POLICY "Enable read access for all users"
  ON students FOR SELECT
  USING (true);

CREATE POLICY "Enable insert for authenticated users"
  ON students FOR INSERT
  WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Enable update for instructors and admins"
  ON students FOR UPDATE
  USING (is_instructor());

CREATE POLICY "Enable delete for admins only"
  ON students FOR DELETE
  USING (is_admin());

-- Attendance policies
CREATE POLICY "Enable read access for all users"
  ON attendance FOR SELECT
  USING (true);

CREATE POLICY "Enable insert for instructors and admins"
  ON attendance FOR INSERT
  WITH CHECK (is_instructor());

CREATE POLICY "Enable update for instructors and admins"
  ON attendance FOR UPDATE
  USING (is_instructor());

-- Storage policies for public bucket (signatures)
CREATE POLICY "Public Access"
  ON storage.objects FOR SELECT
  USING (bucket_id = 'signatures');

CREATE POLICY "Enable insert for authenticated users only"
  ON storage.objects FOR INSERT
  WITH CHECK (
    bucket_id = 'signatures' AND
    auth.role() = 'authenticated' AND
    (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Enable update for authenticated users only"
  ON storage.objects FOR UPDATE
  USING (
    bucket_id = 'signatures' AND
    auth.role() = 'authenticated' AND
    (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Enable delete for authenticated users only"
  ON storage.objects FOR DELETE
  USING (
    bucket_id = 'signatures' AND
    auth.role() = 'authenticated' AND
    (storage.foldername(name))[1] = auth.uid()::text
  );
