-- This is a minimal migration to trigger Supabase types regeneration
-- The system should automatically update the types after any migration

-- Update the RLS policy for backward compatibility
DROP POLICY IF EXISTS "Users can view their own signatures" ON signatures;

-- Create a more specific policy for viewing signatures
CREATE POLICY "Users can view their own signatures"
ON signatures FOR SELECT
USING ((student_id IN ( SELECT students.id
   FROM students
  WHERE (students.id = (( SELECT (auth.uid())::text AS uid))::bigint))) OR (EXISTS ( SELECT 1
   FROM profiles
  WHERE ((profiles.id = auth.uid()) AND (profiles.role = ANY (ARRAY['admin'::text, 'staff'::text]))))));