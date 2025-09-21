-- Fix RLS security issues for AI-related tables
-- Enable RLS on all public tables that need protection

-- Enable RLS on AI-related tables
ALTER TABLE public.global_trained_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.student_signatures ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trained_models ENABLE ROW LEVEL SECURITY;

-- Create policies for global_trained_models (read-only for authenticated users)
CREATE POLICY "Authenticated users can view global models" 
ON public.global_trained_models 
FOR SELECT 
USING (auth.role() = 'authenticated'::text);

CREATE POLICY "Admins can manage global models" 
ON public.global_trained_models 
FOR ALL 
USING ((EXISTS ( SELECT 1 FROM admin WHERE (admin.id = auth.uid()))) OR 
       (EXISTS ( SELECT 1 FROM users WHERE ((users.id = auth.uid()) AND (users.role = ANY (ARRAY['ROTC admin'::text, 'Instructor'::text]))))));

-- Create policies for student_signatures (authenticated users can access)
CREATE POLICY "Authenticated users can manage student signatures" 
ON public.student_signatures 
FOR ALL 
USING (auth.role() = 'authenticated'::text);

-- Create policies for trained_models (authenticated users can access)
CREATE POLICY "Authenticated users can manage trained models" 
ON public.trained_models 
FOR ALL 
USING (auth.role() = 'authenticated'::text);