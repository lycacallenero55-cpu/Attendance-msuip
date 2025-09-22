-- Remove unnecessary columns from students table for cleaner schema
-- These columns are not needed for the signature classification system

ALTER TABLE public.students 
DROP COLUMN IF EXISTS middle_initial,
DROP COLUMN IF EXISTS address, 
DROP COLUMN IF EXISTS birthday,
DROP COLUMN IF EXISTS contact_no,
DROP COLUMN IF EXISTS email;

-- Add sex column if it doesn't exist (keeping existing data if already present)
DO $$ 
BEGIN
    BEGIN
        ALTER TABLE public.students ADD COLUMN sex text;
    EXCEPTION
        WHEN duplicate_column THEN
            -- Column already exists, do nothing
            NULL;
    END;
END $$;

-- Clean up global trained models table - remove unused columns
ALTER TABLE public.global_trained_models 
DROP COLUMN IF EXISTS forged_count;

-- Clean up trained models table - remove unused columns  
ALTER TABLE public.trained_models
DROP COLUMN IF EXISTS forged_count;

-- Update comments to reflect owner identification focus
COMMENT ON TABLE public.global_trained_models IS 'Global signature classification models for owner identification across multiple students';
COMMENT ON TABLE public.trained_models IS 'Individual student signature models for owner identification (legacy - use global models)';
COMMENT ON TABLE public.student_signatures IS 'Student signature samples for training classification models';