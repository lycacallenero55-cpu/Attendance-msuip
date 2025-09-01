-- Remove reason column from excuse_applications table
ALTER TABLE public.excuse_applications 
DROP COLUMN IF EXISTS reason;