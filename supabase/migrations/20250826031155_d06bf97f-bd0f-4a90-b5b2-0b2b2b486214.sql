-- Enable RLS on the excuse_applications table which is missing it
ALTER TABLE excuse_applications ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for excuse_applications
CREATE POLICY "Users can view excuse applications"
ON excuse_applications FOR SELECT
USING (auth.role() = 'authenticated'::text);

CREATE POLICY "Users can create excuse applications"
ON excuse_applications FOR INSERT
WITH CHECK (auth.role() = 'authenticated'::text);

CREATE POLICY "Users can update excuse applications"
ON excuse_applications FOR UPDATE
USING (auth.role() = 'authenticated'::text);

CREATE POLICY "Users can delete excuse applications"
ON excuse_applications FOR DELETE
USING (auth.role() = 'authenticated'::text);