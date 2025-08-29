-- Step 1: Update role values in profiles table
UPDATE profiles SET role = 'staff' WHERE role = 'instructor';
UPDATE profiles SET role = 'ssg_officer' WHERE role = 'user';

-- Step 2: Add created_by column to sessions table
ALTER TABLE sessions ADD COLUMN created_by_user_id UUID REFERENCES profiles(id);

-- Step 3: For backward compatibility, set existing sessions to NULL
-- (Will be displayed as 'System' in the UI)

-- Step 4: Remove location column from sessions table
ALTER TABLE sessions DROP COLUMN IF EXISTS location;

-- Step 5: Remove instructor column from sessions table (replaced by created_by_user_id)
ALTER TABLE sessions DROP COLUMN IF EXISTS instructor;

-- Step 6: Update RLS policies to work with new role names
DROP POLICY IF EXISTS "Users can view signatures" ON signatures;
DROP POLICY IF EXISTS "Users can insert their own signatures" ON signatures;
DROP POLICY IF EXISTS "Users can update their own signatures" ON signatures;
DROP POLICY IF EXISTS "Users can delete their own signatures" ON signatures;

-- Create updated RLS policies for signatures
CREATE POLICY "Users can view signatures"
ON signatures FOR SELECT
USING ((student_id IN ( SELECT students.id
   FROM students
  WHERE (students.id = (( SELECT (auth.uid())::text AS uid))::bigint))) OR (EXISTS ( SELECT 1
   FROM profiles
  WHERE ((profiles.id = auth.uid()) AND (profiles.role = ANY (ARRAY['admin'::text, 'staff'::text]))))));

CREATE POLICY "Users can insert their own signatures"
ON signatures FOR INSERT
WITH CHECK ((student_id IN ( SELECT students.id
   FROM students
  WHERE (students.id = (( SELECT (auth.uid())::text AS uid))::bigint))) OR (EXISTS ( SELECT 1
   FROM profiles
  WHERE ((profiles.id = auth.uid()) AND (profiles.role = ANY (ARRAY['admin'::text, 'staff'::text]))))));

CREATE POLICY "Users can update their own signatures"
ON signatures FOR UPDATE
USING ((student_id IN ( SELECT students.id
   FROM students
  WHERE (students.id = (( SELECT (auth.uid())::text AS uid))::bigint))) OR (EXISTS ( SELECT 1
   FROM profiles
  WHERE ((profiles.id = auth.uid()) AND (profiles.role = ANY (ARRAY['admin'::text, 'staff'::text]))))));

CREATE POLICY "Users can delete their own signatures"
ON signatures FOR DELETE
USING ((student_id IN ( SELECT students.id
   FROM students
  WHERE (students.id = (( SELECT (auth.uid())::text AS uid))::bigint))) OR (EXISTS ( SELECT 1
   FROM profiles
  WHERE ((profiles.id = auth.uid()) AND (profiles.role = ANY (ARRAY['admin'::text, 'staff'::text]))))));

-- Update other RLS policies that reference instructor role
DROP POLICY IF EXISTS "Admins can manage academic years" ON academic_years;
CREATE POLICY "Admins can manage academic years"
ON academic_years FOR ALL
USING (EXISTS ( SELECT 1
   FROM profiles
  WHERE ((profiles.id = auth.uid()) AND (profiles.role = 'admin'::text))));

DROP POLICY IF EXISTS "Admins can manage semesters" ON semesters;
CREATE POLICY "Admins can manage semesters"
ON semesters FOR ALL
USING (EXISTS ( SELECT 1
   FROM profiles
  WHERE ((profiles.id = auth.uid()) AND (profiles.role = 'admin'::text))));

-- Update helper functions that check roles
DROP FUNCTION IF EXISTS public.is_instructor();
CREATE OR REPLACE FUNCTION public.is_staff()
RETURNS boolean
LANGUAGE sql
SECURITY DEFINER
AS $function$
  SELECT EXISTS (
    SELECT 1 FROM profiles 
    WHERE id = auth.uid() AND role IN ('admin', 'staff')
  );
$function$;

-- Create role label mapping function for UI display
CREATE OR REPLACE FUNCTION public.get_role_label(role_name text)
RETURNS text
LANGUAGE sql
IMMUTABLE
AS $function$
  SELECT CASE 
    WHEN role_name = 'admin' THEN 'Admin'
    WHEN role_name = 'staff' THEN 'Staff'
    WHEN role_name = 'ssg_officer' THEN 'SSG Officer'
    ELSE 'Unknown'
  END;
$function$;