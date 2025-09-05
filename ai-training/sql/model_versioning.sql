-- Model Versioning Schema Updates
-- Run these SQL commands in your Supabase SQL editor

-- Add versioning columns to trained_models table
ALTER TABLE public.trained_models 
ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1,
ADD COLUMN IF NOT EXISTS parent_model_id BIGINT,
ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true,
ADD COLUMN IF NOT EXISTS version_notes TEXT,
ADD COLUMN IF NOT EXISTS performance_metrics JSONB;

-- Create index for better query performance
CREATE INDEX IF NOT EXISTS idx_trained_models_student_version 
ON public.trained_models(student_id, version);

CREATE INDEX IF NOT EXISTS idx_trained_models_active 
ON public.trained_models(student_id, is_active) 
WHERE is_active = true;

-- Create model_versions table for detailed version tracking
CREATE TABLE IF NOT EXISTS public.model_versions (
    id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES public.trained_models(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by TEXT,
    version_notes TEXT,
    performance_metrics JSONB,
    model_artifacts JSONB, -- Store paths to different model files
    is_active BOOLEAN DEFAULT false,
    UNIQUE(model_id, version)
);

-- Create audit log table for model changes
CREATE TABLE IF NOT EXISTS public.model_audit_log (
    id BIGSERIAL PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES public.trained_models(id) ON DELETE CASCADE,
    action TEXT NOT NULL, -- 'created', 'updated', 'activated', 'deactivated', 'deleted'
    old_values JSONB,
    new_values JSONB,
    performed_by TEXT,
    performed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT
);

-- Create A/B testing table
CREATE TABLE IF NOT EXISTS public.model_ab_tests (
    id BIGSERIAL PRIMARY KEY,
    student_id BIGINT NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
    model_a_id BIGINT NOT NULL REFERENCES public.trained_models(id) ON DELETE CASCADE,
    model_b_id BIGINT NOT NULL REFERENCES public.trained_models(id) ON DELETE CASCADE,
    test_name TEXT NOT NULL,
    description TEXT,
    start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_date TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    traffic_split DECIMAL(3,2) DEFAULT 0.5, -- 0.5 = 50/50 split
    results JSONB,
    created_by TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create verification results table for A/B testing
CREATE TABLE IF NOT EXISTS public.verification_results (
    id BIGSERIAL PRIMARY KEY,
    student_id BIGINT NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
    model_id BIGINT NOT NULL REFERENCES public.trained_models(id) ON DELETE CASCADE,
    ab_test_id BIGINT REFERENCES public.model_ab_tests(id) ON DELETE SET NULL,
    test_signature_path TEXT,
    verification_result JSONB, -- Store full verification response
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_model_versions_model_id ON public.model_versions(model_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_active ON public.model_versions(model_id, is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_model_audit_log_model_id ON public.model_audit_log(model_id);
CREATE INDEX IF NOT EXISTS idx_model_ab_tests_student_id ON public.model_ab_tests(student_id);
CREATE INDEX IF NOT EXISTS idx_verification_results_student_id ON public.verification_results(student_id);
CREATE INDEX IF NOT EXISTS idx_verification_results_model_id ON public.verification_results(model_id);
CREATE INDEX IF NOT EXISTS idx_verification_results_ab_test_id ON public.verification_results(ab_test_id);

-- Add triggers for automatic versioning
CREATE OR REPLACE FUNCTION update_model_version()
RETURNS TRIGGER AS $$
BEGIN
    -- If this is a new model, set version to 1
    IF TG_OP = 'INSERT' THEN
        NEW.version := 1;
        NEW.is_active := true;
    END IF;
    
    -- If this is an update and status changed to completed, create a new version
    IF TG_OP = 'UPDATE' AND OLD.status != 'completed' AND NEW.status = 'completed' THEN
        -- Deactivate previous versions
        UPDATE public.trained_models 
        SET is_active = false 
        WHERE student_id = NEW.student_id AND id != NEW.id;
        
        -- Create version record
        INSERT INTO public.model_versions (
            model_id, version, created_at, version_notes, 
            performance_metrics, model_artifacts, is_active
        ) VALUES (
            NEW.id, NEW.version, NOW(), NEW.version_notes,
            NEW.performance_metrics, 
            jsonb_build_object(
                'model_path', NEW.model_path,
                'embedding_model_path', NEW.embedding_model_path
            ),
            true
        );
        
        -- Log the action
        INSERT INTO public.model_audit_log (
            model_id, action, new_values, performed_at
        ) VALUES (
            NEW.id, 'version_created', 
            jsonb_build_object('version', NEW.version, 'status', NEW.status),
            NOW()
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
DROP TRIGGER IF EXISTS trigger_update_model_version ON public.trained_models;
CREATE TRIGGER trigger_update_model_version
    AFTER INSERT OR UPDATE ON public.trained_models
    FOR EACH ROW
    EXECUTE FUNCTION update_model_version();

-- Add RLS policies for security
ALTER TABLE public.model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_ab_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.verification_results ENABLE ROW LEVEL SECURITY;

-- Create policies (adjust based on your auth setup)
CREATE POLICY "Allow authenticated access to model_versions" ON public.model_versions
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated access to model_audit_log" ON public.model_audit_log
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated access to model_ab_tests" ON public.model_ab_tests
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated access to verification_results" ON public.verification_results
    FOR ALL USING (auth.role() = 'authenticated');