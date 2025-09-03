import Layout from "@/components/Layout";
import { useAuth } from "@/hooks/useAuth";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { User, Mail, Calendar, Shield, Edit, Eye, EyeOff } from "lucide-react";
import { useState, useEffect } from "react";
import { supabase } from "@/lib/supabase";
import { toast } from "sonner";

interface UserProfile {
  id: string;
  role: string;
  first_name?: string;
  last_name?: string;
  updated_at?: string;
}

const Profile = () => {
  const { user } = useAuth();
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  
  // Edit Profile Dialog State
  const [editProfileOpen, setEditProfileOpen] = useState(false);
  const [editFirstName, setEditFirstName] = useState('');
  const [editLastName, setEditLastName] = useState('');
  const [isUpdatingProfile, setIsUpdatingProfile] = useState(false);
  
  // Change Password Dialog State
  const [changePasswordOpen, setChangePasswordOpen] = useState(false);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isChangingPassword, setIsChangingPassword] = useState(false);

  useEffect(() => {
    const fetchUserProfile = async () => {
      if (!user) return;

      try {
        // Try admin first
        const { data: adminData } = await supabase
          .from('admin')
          .select('*')
          .eq('id', user.id)
          .maybeSingle();
        if (adminData) {
          // Ensure role is present for admin accounts so UI shows Admin
          setUserProfile({ ...adminData, role: 'admin' });
        } else {
          const { data: userData } = await supabase
            .from('users')
            .select('*')
            .eq('id', user.id)
            .maybeSingle();
          setUserProfile(userData);
        }
      } catch (error) {
        console.error('Error fetching user profile:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchUserProfile();
  }, [user]);

  // Handle Edit Profile
  const handleEditProfile = () => {
    setEditFirstName(userProfile?.first_name || '');
    setEditLastName(userProfile?.last_name || '');
    setEditProfileOpen(true);
  };

  const handleUpdateProfile = async () => {
    if (!user || !userProfile) return;

    setIsUpdatingProfile(true);
    try {
      // Update to the correct table
      const targetTable = userProfile?.role ? 'users' : 'admin';
      const { error } = await supabase
        .from(targetTable)
        .update({
          first_name: editFirstName.trim(),
          last_name: editLastName.trim(),
          updated_at: new Date().toISOString()
        })
        .eq('id', user.id);

      if (error) throw error;

      // Update local state
      setUserProfile(prev => prev ? {
        ...prev,
        first_name: editFirstName.trim(),
        last_name: editLastName.trim(),
        updated_at: new Date().toISOString()
      } : null);

      toast.success('Profile updated successfully!');
      setEditProfileOpen(false);
    } catch (error) {
      console.error('Error updating profile:', error);
      toast.error('Failed to update profile. Please try again.');
    } finally {
      setIsUpdatingProfile(false);
    }
  };

  // Handle Change Password
  const handleChangePassword = () => {
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setChangePasswordOpen(true);
  };

  const handleUpdatePassword = async () => {
    if (!user) return;

    // Validation
    if (!currentPassword.trim()) {
      toast.error('Please enter your current password');
      return;
    }

    if (!newPassword.trim()) {
      toast.error('Please enter a new password');
      return;
    }

    if (newPassword !== confirmPassword) {
      toast.error('New passwords do not match');
      return;
    }

    if (newPassword.length < 6) {
      toast.error('New password must be at least 6 characters long');
      return;
    }

    if (currentPassword === newPassword) {
      toast.error('New password must be different from current password');
      return;
    }

    setIsChangingPassword(true);
    try {
      // First, verify the current password by attempting to sign in
      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: user.email!,
        password: currentPassword
      });

      if (signInError) {
        toast.error('Current password is incorrect');
        return;
      }

      // If current password is correct, update to new password
      const { error } = await supabase.auth.updateUser({
        password: newPassword
      });

      if (error) throw error;

      toast.success('Password changed successfully!');
      setChangePasswordOpen(false);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (error) {
      console.error('Error changing password:', error);
      toast.error('Failed to change password. Please try again.');
    } finally {
      setIsChangingPassword(false);
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="text-muted-foreground">Loading profile...</div>
        </div>
      </Layout>
    );
  }

  if (!user) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="text-muted-foreground">No user found</div>
        </div>
      </Layout>
    );
  }

  // Get user initials from email
  const getInitials = (email: string) => {
    return email
      .split("@")[0]
      .split(".")
      .map((n) => n[0])
      .join("")
      .toUpperCase();
  };

  // Normalize and display role name for both legacy and new roles
  const getRoleDisplayName = (role: string) => {
    const r = (role || '').toLowerCase().replace(/_/g, ' ');
    if (r === 'admin') return 'Administrator';
    if (r === 'instructor' || r === 'staff') return 'Instructor';
    if (r === 'ssg officer' || r === 'ssg_officer') return 'SSG Officer';
    if (r === 'rotc admin') return 'ROTC Admin';
    if (r === 'rotc officer') return 'ROTC Officer';
    if (r === 'student') return 'Student';
    return 'User';
  };

  // Get role badge variant
  const getRoleBadgeVariant = (role: string) => {
    const r = (role || '').toLowerCase().replace(/_/g, ' ');
    if (r === 'admin') return 'destructive';
    if (r === 'instructor' || r === 'staff') return 'default';
    if (r === 'ssg officer' || r === 'rotc admin' || r === 'rotc officer') return 'secondary';
    if (r === 'student') return 'secondary';
    return 'outline';
  };

  return (
    <Layout>
      <div className="px-6 py-4 space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-lg font-bold tracking-tight">PROFILE</h1>
          <p className="text-sm text-muted-foreground">
            Manage your account information and preferences
          </p>
        </div>

        {/* Profile Information Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Profile Information
            </CardTitle>
            <CardDescription>
              Your personal account details
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Avatar and Basic Info */}
            <div className="flex items-center gap-4">
              <Avatar className="h-16 w-16">
                <AvatarImage src={user.user_metadata?.avatar_url} />
                <AvatarFallback className="bg-primary/10 text-primary text-lg">
                  {user.email ? getInitials(user.email) : <User className="h-6 w-6" />}
                </AvatarFallback>
              </Avatar>
                             <div className="space-y-1">
                 <h3 className="text-lg font-semibold">
                   {userProfile?.first_name && userProfile?.last_name 
                     ? `${userProfile.first_name} ${userProfile.last_name}`
                     : user.user_metadata?.full_name || 'User'}
                 </h3>
                 <p className="text-sm text-muted-foreground">
                   {user.email}
                 </p>
                <Badge variant={getRoleBadgeVariant(userProfile?.role || 'user')}>
                  {getRoleDisplayName(userProfile?.role || 'user')}
                </Badge>
              </div>
            </div>

            <Separator />

            {/* Account Details */}
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <Mail className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">Email</p>
                  <p className="text-sm text-muted-foreground">{user.email}</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <Shield className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">Role</p>
                  <p className="text-sm text-muted-foreground">
                    {getRoleDisplayName(userProfile?.role || 'user')}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <Calendar className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">Member Since</p>
                  <p className="text-sm text-muted-foreground">
                    {user.created_at ? new Date(user.created_at).toLocaleDateString() : 'Unknown'}
                  </p>
                </div>
              </div>
            </div>

            <Separator />

                         {/* Account Actions */}
             <div className="flex gap-3 justify-center">
               <Button variant="outline" size="sm" onClick={handleEditProfile}>
                 <Edit className="h-4 w-4 mr-2" />
                 Edit Profile
               </Button>
               
               <Button variant="outline" size="sm" onClick={handleChangePassword}>
                 <Shield className="h-4 w-4 mr-2" />
                 Change Password
               </Button>
             </div>
          </CardContent>
        </Card>
      </div>

      {/* Edit Profile Dialog */}
      <Dialog open={editProfileOpen} onOpenChange={setEditProfileOpen}>
        <DialogContent className="!max-w-md w-[90vw] mx-auto rounded-lg">
          <DialogHeader>
            <DialogTitle>Edit Profile</DialogTitle>
            <DialogDescription>
              Update your first name and last name
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="firstName">First Name</Label>
              <Input
                id="firstName"
                value={editFirstName}
                onChange={(e) => setEditFirstName(e.target.value)}
                placeholder="Enter your first name"
              />
            </div>
            <div>
              <Label htmlFor="lastName">Last Name</Label>
              <Input
                id="lastName"
                value={editLastName}
                onChange={(e) => setEditLastName(e.target.value)}
                placeholder="Enter your last name"
              />
            </div>
          </div>
          <DialogFooter className="!flex !flex-row !justify-end gap-2 sm:gap-3">
            <Button 
              variant="outline" 
              onClick={() => setEditProfileOpen(false)}
              className="flex-1 sm:flex-none sm:min-w-[80px]"
            >
              Cancel
            </Button>
            <Button 
              onClick={handleUpdateProfile} 
              disabled={isUpdatingProfile}
              className="flex-1 sm:flex-none sm:min-w-[80px]"
            >
              {isUpdatingProfile ? 'Updating...' : 'Update Profile'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Change Password Dialog */}
      <Dialog open={changePasswordOpen} onOpenChange={setChangePasswordOpen}>
        <DialogContent className="!max-w-md w-[90vw] mx-auto rounded-lg">
          <DialogHeader>
            <DialogTitle>Change Password</DialogTitle>
            <DialogDescription>
              Enter your current password and choose a new password
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="currentPassword">Current Password</Label>
              <div className="relative">
                <Input
                  id="currentPassword"
                  type={showCurrentPassword ? "text" : "password"}
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  placeholder="Enter your current password"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                  onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                >
                  {showCurrentPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
            <div>
              <Label htmlFor="newPassword">New Password</Label>
              <div className="relative">
                <Input
                  id="newPassword"
                  type={showNewPassword ? "text" : "password"}
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="Enter your new password"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                >
                  {showNewPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
            <div>
              <Label htmlFor="confirmPassword">Confirm New Password</Label>
              <div className="relative">
                <Input
                  id="confirmPassword"
                  type={showConfirmPassword ? "text" : "password"}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Confirm your new password"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                >
                  {showConfirmPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
          </div>
          <DialogFooter className="!flex !flex-row !justify-end gap-2 sm:gap-3">
            <Button 
              variant="outline" 
              onClick={() => setChangePasswordOpen(false)}
              className="flex-1 sm:flex-none sm:min-w-[80px]"
            >
              Cancel
            </Button>
            <Button 
              onClick={handleUpdatePassword} 
              disabled={isChangingPassword}
              className="flex-1 sm:flex-none sm:min-w-[80px]"
            >
              {isChangingPassword ? 'Changing...' : 'Change Password'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Layout>
  );
};

export default Profile;
