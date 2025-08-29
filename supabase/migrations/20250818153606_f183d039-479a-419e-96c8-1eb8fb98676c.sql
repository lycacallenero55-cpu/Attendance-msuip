-- Add account status and approval system to profiles table
ALTER TABLE public.profiles 
ADD COLUMN status text DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'inactive', 'suspended')),
ADD COLUMN approved_by uuid REFERENCES auth.users(id),
ADD COLUMN approved_at timestamp with time zone,
ADD COLUMN rejected_by uuid REFERENCES auth.users(id),
ADD COLUMN rejected_at timestamp with time zone;

-- Create sign up function
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
BEGIN
  INSERT INTO public.profiles (id, email, first_name, last_name, role, status)
  VALUES (
    NEW.id,
    NEW.email,
    NEW.raw_user_meta_data ->> 'first_name',
    NEW.raw_user_meta_data ->> 'last_name', 
    COALESCE(NEW.raw_user_meta_data ->> 'role', 'user'),
    'pending'
  );
  RETURN NEW;
END;
$$;

-- Create trigger for new user signup
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Create function to approve/reject users
CREATE OR REPLACE FUNCTION public.approve_user(user_id uuid, approver_id uuid)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  UPDATE public.profiles
  SET 
    status = 'active',
    approved_by = approver_id,
    approved_at = now()
  WHERE id = user_id AND status = 'pending';
  
  IF FOUND THEN
    RETURN jsonb_build_object('success', true, 'message', 'User approved successfully');
  ELSE
    RETURN jsonb_build_object('success', false, 'message', 'User not found or already processed');
  END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.reject_user(user_id uuid, rejector_id uuid)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  UPDATE public.profiles
  SET 
    status = 'inactive',
    rejected_by = rejector_id,
    rejected_at = now()
  WHERE id = user_id AND status = 'pending';
  
  IF FOUND THEN
    RETURN jsonb_build_object('success', true, 'message', 'User rejected successfully');
  ELSE
    RETURN jsonb_build_object('success', false, 'message', 'User not found or already processed');
  END IF;
END;
$$;