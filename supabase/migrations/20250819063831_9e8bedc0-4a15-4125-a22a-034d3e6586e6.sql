-- Fix excuse_applications table - remove excuse_image_url column and use documentation_url instead
ALTER TABLE excuse_applications DROP COLUMN IF EXISTS excuse_image_url;

-- Ensure documentation_url column exists and is properly configured
ALTER TABLE excuse_applications 
  ALTER COLUMN documentation_url TYPE text,
  ALTER COLUMN documentation_url DROP NOT NULL;