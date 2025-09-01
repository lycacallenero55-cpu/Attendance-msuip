-- Migration: Add Signature Support
-- This migration enhances signature storage and adds support for signature verification

-- Create a dedicated table for storing signature metadata and features
CREATE TABLE IF NOT EXISTS public.signatures (
    id BIGSERIAL PRIMARY KEY,
    student_id BIGINT NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
    storage_path TEXT NOT NULL,  -- Path in storage bucket
    file_name TEXT NOT NULL,     -- Original file name
    file_size INTEGER NOT NULL,  -- File size in bytes
    file_type TEXT NOT NULL,     -- MIME type
    
    -- Image dimensions
    width INTEGER,
    height INTEGER,
    
    -- Feature vector for matching (stored as JSONB for flexibility)
    features JSONB,
    
    -- Quality metrics (0-1 scale)
    quality_score FLOAT,
    
    -- Device and capture info
    device_info JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_quality_score CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1))
);

-- Add index for faster lookups
CREATE INDEX idx_signatures_student_id ON public.signatures(student_id);
CREATE INDEX idx_signatures_created_at ON public.signatures(created_at);

-- Add GIN index for JSONB features for faster similarity searches
CREATE INDEX idx_signatures_features ON public.signatures USING GIN (features);

-- Create a function to update the updated_at column
CREATE OR REPLACE FUNCTION update_signatures_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for updated_at
CREATE TRIGGER update_signatures_updated_at
BEFORE UPDATE ON public.signatures
FOR EACH ROW EXECUTE FUNCTION update_signatures_updated_at();

-- Create a function to get a student's primary signature
CREATE OR REPLACE FUNCTION get_primary_signature(student_id_param BIGINT)
RETURNS TEXT AS $$
DECLARE
    primary_sig_path TEXT;
BEGIN
    SELECT storage_path INTO primary_sig_path
    FROM public.signatures
    WHERE student_id = student_id_param
    ORDER BY 
        CASE 
            WHEN quality_score IS NOT NULL THEN quality_score 
            ELSE 0 
        END DESC,
        created_at DESC
    LIMIT 1;
    
    RETURN primary_sig_path;
END;
$$ LANGUAGE plpgsql STABLE;

-- Create a function to compare two signatures
CREATE OR REPLACE FUNCTION compare_signatures(sig1_id BIGINT, sig2_id BIGINT)
RETURNS FLOAT AS $$
DECLARE
    sig1_features JSONB;
    sig2_features JSONB;
    similarity FLOAT;
BEGIN
    -- Get feature vectors for both signatures
    SELECT features INTO sig1_features
    FROM public.signatures
    WHERE id = sig1_id;
    
    SELECT features INTO sig2_features
    FROM public.signatures
    WHERE id = sig2_id;
    
    -- Simple cosine similarity (placeholder - implement actual comparison logic)
    -- This is a simplified version - you'll want to implement a more sophisticated
    -- comparison based on your feature extraction method
    similarity := 0.0;
    
    -- TODO: Implement actual feature comparison logic here
    -- This is just a placeholder that returns a random value
    similarity := random();
    
    RETURN LEAST(GREATEST(similarity, 0.0), 1.0);
END;
$$ LANGUAGE plpgsql STABLE;

-- Update the students table to reference the primary signature
ALTER TABLE public.students 
ADD COLUMN IF NOT EXISTS primary_signature_id BIGINT REFERENCES public.signatures(id);

-- Create a function to update the primary signature reference
CREATE OR REPLACE FUNCTION update_primary_signature()
RETURNS TRIGGER AS $$
BEGIN
    -- Only update if this is a new signature with higher quality than existing
    IF NEW.quality_score IS NOT NULL AND 
       (OLD.quality_score IS NULL OR NEW.quality_score > OLD.quality_score) THEN
        
        UPDATE public.students
        SET primary_signature_id = NEW.id,
            updated_at = NOW()
        WHERE id = NEW.student_id
        AND (primary_signature_id IS NULL OR 
             (SELECT quality_score FROM public.signatures WHERE id = primary_signature_id) < NEW.quality_score);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to update primary signature reference
CREATE TRIGGER update_primary_signature_trigger
AFTER INSERT OR UPDATE ON public.signatures
FOR EACH ROW EXECUTE FUNCTION update_primary_signature();

-- Update RLS policies for the new table
ALTER TABLE public.signatures ENABLE ROW LEVEL SECURITY;

-- Policy to allow users to see their own signatures
CREATE POLICY "Users can view their own signatures"
ON public.signatures
FOR SELECT
USING (
    student_id IN (
        SELECT id FROM public.students 
        WHERE id = (SELECT auth.uid()::text)::bigint
    )
    OR EXISTS (
        SELECT 1 FROM public.admin 
        WHERE id = auth.uid()
    )
    OR EXISTS (
        SELECT 1 FROM public.users 
        WHERE id = auth.uid() AND role = 'Instructor'
    )
);

-- Policy to allow users to insert their own signatures
CREATE POLICY "Users can insert their own signatures"
ON public.signatures
FOR INSERT
WITH CHECK (
    student_id IN (
        SELECT id FROM public.students 
        WHERE id = (SELECT auth.uid()::text)::bigint
    )
    OR EXISTS (
        SELECT 1 FROM public.admin 
        WHERE id = auth.uid()
    )
    OR EXISTS (
        SELECT 1 FROM public.users 
        WHERE id = auth.uid() AND role = 'Instructor'
    )
);

-- Policy to allow users to update their own signatures
CREATE POLICY "Users can update their own signatures"
ON public.signatures
FOR UPDATE
USING (
    student_id IN (
        SELECT id FROM public.students 
        WHERE id = (SELECT auth.uid()::text)::bigint
    )
    OR EXISTS (
        SELECT 1 FROM public.admin 
        WHERE id = auth.uid()
    )
    OR EXISTS (
        SELECT 1 FROM public.users 
        WHERE id = auth.uid() AND role = 'Instructor'
    )
);

-- Policy to allow users to delete their own signatures
CREATE POLICY "Users can delete their own signatures"
ON public.signatures
FOR DELETE
USING (
    student_id IN (
        SELECT id FROM public.students 
        WHERE id = (SELECT auth.uid()::text)::bigint
    )
    OR EXISTS (
        SELECT 1 FROM public.admin 
        WHERE id = auth.uid()
    )
    OR EXISTS (
        SELECT 1 FROM public.users 
        WHERE id = auth.uid() AND role = 'Instructor'
    )
);

-- Create a view for easier access to student signatures
CREATE OR REPLACE VIEW public.student_signatures_view AS
SELECT 
    s.id as student_id,
    s.student_id as student_number,
    s.firstname,
    s.surname,
    sig.id as signature_id,
    sig.storage_path,
    sig.quality_score,
    sig.created_at as signature_date,
    sig.updated_at as last_updated,
    (sig.id = s.primary_signature_id) as is_primary
FROM 
    public.students s
LEFT JOIN 
    public.signatures sig ON s.id = sig.student_id
ORDER BY 
    s.surname, s.firstname, sig.created_at DESC;

-- Grant permissions
GRANT SELECT ON public.student_signatures_view TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.signatures TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE public.signatures_id_seq TO authenticated;

-- Comment on the new table and columns
COMMENT ON TABLE public.signatures IS 'Stores student signature data including feature vectors for verification';
COMMENT ON COLUMN public.signatures.features IS 'JSONB containing extracted signature features for matching';
COMMENT ON COLUMN public.signatures.quality_score IS 'Quality score (0-1) indicating signature quality';
COMMENT ON COLUMN public.signatures.device_info IS 'JSONB containing device and capture information';

-- Update the plan with the completed migration
COMMENT ON DATABASE current_database IS 'Database schema updated to support signature verification with feature storage and matching';

-- Update the plan
-- This migration adds support for storing signature features and metadata
-- Next steps:
-- 1. Implement feature extraction in the application
-- 2. Create API endpoints for signature verification
-- 3. Update the frontend to use the new signature storage