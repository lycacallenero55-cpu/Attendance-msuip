import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogDescription } from "@/components/ui/dialog";
import { Plus, X } from "lucide-react";
import { toast } from "sonner";
import { supabase } from "@/lib/supabase";
import { useUnsavedChanges } from "@/hooks/useUnsavedChanges";
import { UnsavedChangesDialog } from "@/components/UnsavedChangesDialog";

interface StudentFormData {
  surname: string;
  middle_initial: string;
  firstname: string;
  student_id: string;
  program: string;
  year: string;
  section: string;
  sex: string;
  address: string;
  birthday: string;
  contact_no: string;
  email: string;
}

interface StudentFormProps {
  onStudentAdded: () => void;
}

const StudentForm: React.FC<StudentFormProps> = ({ onStudentAdded }) => {
  const [open, setOpen] = useState(false);
  const [formData, setFormData] = useState<StudentFormData>({
    surname: '',
    middle_initial: '',
    firstname: '',
    student_id: '',
    program: '',
    year: '',
    section: '',
    sex: '',
    address: '',
    birthday: '',
    contact_no: '',
    email: ''
  });

  const {
    showConfirmDialog,
    markAsChanged,
    markAsSaved,
    handleClose,
    confirmClose,
    cancelClose,
    handleOpenChange,
  } = useUnsavedChanges({
    onClose: () => setOpen(false),
    enabled: open,
  });

  const handleInputChange = (field: keyof StudentFormData, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    markAsChanged();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const { data, error } = await supabase
        .from('students')
        .insert([{
          ...formData,
          // Ensure all required fields are present
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }])
        .select();

      if (error) {
        throw error;
      }

      if (data && data.length > 0) {
        toast.success('Student added successfully!');
        markAsSaved();
        // Reset form
        setFormData({
          surname: '',
          middle_initial: '',
          firstname: '',
          student_id: '',
          program: '',
          year: '',
          section: '',
          sex: '',
          address: '',
          birthday: '',
          contact_no: '',
          email: ''
        });
        setOpen(false);
        onStudentAdded();
      } else {
        throw new Error('No data returned from database');
      }
    } catch (error) {
      console.error('Error adding student:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to add student');
    }
  };

  return (
    <>
      <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button className="bg-gradient-primary shadow-glow">
          <Plus className="w-4 h-4 mr-2" />
          Add Student
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex justify-between items-start">
            <div>
              <DialogTitle className="text-2xl font-bold text-education-navy">
                Add New Student
              </DialogTitle>
              <DialogDescription>
                Fill in the student's information. Fields marked with * are required.
              </DialogDescription>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleClose}
              className="h-8 w-8 text-muted-foreground hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="surname">Surname *</Label>
              <Input
                id="surname"
                value={formData.surname}
                onChange={(e) => handleInputChange('surname', e.target.value)}
                placeholder="Enter surname"
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="middle_initial">M.I.</Label>
              <Input
                id="middle_initial"
                value={formData.middle_initial}
                onChange={(e) => handleInputChange('middle_initial', e.target.value)}
                placeholder="M.I."
                maxLength={10}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="firstname">Firstname *</Label>
              <Input
                id="firstname"
                value={formData.firstname}
                onChange={(e) => handleInputChange('firstname', e.target.value)}
                placeholder="Enter firstname"
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="student_id">Student ID *</Label>
              <Input
                id="student_id"
                value={formData.student_id}
                onChange={(e) => handleInputChange('student_id', e.target.value)}
                placeholder="Enter student ID"
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="program">Program *</Label>
              <Input
                id="program"
                value={formData.program}
                onChange={(e) => handleInputChange('program', e.target.value)}
                placeholder="Enter program"
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="year">Year *</Label>
              <Select value={formData.year} onValueChange={(value) => handleInputChange('year', value)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select year" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1st">1st Year</SelectItem>
                  <SelectItem value="2nd">2nd Year</SelectItem>
                  <SelectItem value="3rd">3rd Year</SelectItem>
                  <SelectItem value="4th">4th Year</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="section">Section *</Label>
              <Input
                id="section"
                value={formData.section}
                onChange={(e) => handleInputChange('section', e.target.value)}
                placeholder="Enter section"
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="sex">Sex *</Label>
              <Select value={formData.sex} onValueChange={(value) => handleInputChange('sex', value)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select sex" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Male">Male</SelectItem>
                  <SelectItem value="Female">Female</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="address">Address</Label>
            <Input
              id="address"
              value={formData.address}
              onChange={(e) => handleInputChange('address', e.target.value)}
              placeholder="Enter address"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="birthday">Birthday</Label>
              <Input
                id="birthday"
                type="date"
                value={formData.birthday}
                onChange={(e) => handleInputChange('birthday', e.target.value)}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="contact_no">Contact No.</Label>
              <Input
                id="contact_no"
                value={formData.contact_no}
                onChange={(e) => handleInputChange('contact_no', e.target.value)}
                placeholder="Enter contact number"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              value={formData.email}
              onChange={(e) => handleInputChange('email', e.target.value)}
              placeholder="Enter email address"
            />
          </div>

          <div className="flex justify-end gap-3 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={handleClose}
            >
              Cancel
            </Button>
            <Button type="submit" className="bg-gradient-primary shadow-glow">
              Add Student
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>

    <UnsavedChangesDialog
      open={showConfirmDialog}
      onConfirm={confirmClose}
      onCancel={cancelClose}
    />
  </>
  );
};

export default StudentForm; 