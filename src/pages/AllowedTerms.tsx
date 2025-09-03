import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Calendar, CalendarDays, Plus, Search, Trash2 } from "lucide-react";
import { format } from "date-fns";
import { useToast } from "@/components/ui/use-toast";
import { supabase } from "@/lib/supabase";
import Layout from "@/components/Layout";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

 type AllowedTerm = {
  id: string;
  academic_year: string;
  semester: string;
  start_date: string; // ISO date
  end_date: string;   // ISO date
  created_at: string;
};

const AllowedTerms = () => {
  const { toast } = useToast();
  const [terms, setTerms] = useState<AllowedTerm[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchTerm, setSearchTerm] = useState<string>('');

  const [formData, setFormData] = useState<{
    academic_year: string;
    semester: string;
    start_date: string;
    end_date: string;
  }>({ academic_year: '', semester: '', start_date: '', end_date: '' });
  const [isFormOpen, setIsFormOpen] = useState<boolean>(false);

  useEffect(() => {
    fetchAllowedTerms();
  }, []);

  const fetchAllowedTerms = async () => {
    try {
      setLoading(true);
      const { data, error } = await supabase
        .from('allowed_terms')
        .select('*')
        .order('created_at', { ascending: false });
      if (error) throw error;
      setTerms(data || []);
    } catch (error) {
      console.error('Error fetching allowed terms:', error);
      toast({ title: 'Error', description: 'Failed to fetch allowed terms', variant: 'destructive' });
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    try {
      if (!formData.academic_year || !formData.semester || !formData.start_date || !formData.end_date) {
        toast({ title: 'Missing fields', description: 'Please fill in all fields.' });
        return;
      }

      const { error } = await supabase.from('allowed_terms').insert([
        {
          academic_year: formData.academic_year.trim(),
          semester: formData.semester.trim(),
          start_date: formData.start_date,
          end_date: formData.end_date,
        },
      ]);
      if (error) throw error;

      toast({ title: 'Success', description: 'Allowed term added successfully.' });
      setFormData({ academic_year: '', semester: '', start_date: '', end_date: '' });
      setIsFormOpen(false);
      fetchAllowedTerms();
    } catch (error) {
      console.error('Error creating allowed term:', error);
      toast({ title: 'Error', description: 'Failed to add allowed term', variant: 'destructive' });
    }
  };

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this allowed term?')) return;
    try {
      const { error } = await supabase.from('allowed_terms').delete().eq('id', id);
      if (error) throw error;
      toast({ title: 'Deleted', description: 'Allowed term removed.' });
      fetchAllowedTerms();
    } catch (error) {
      console.error('Delete error:', error);
      toast({ title: 'Error', description: 'Failed to delete allowed term', variant: 'destructive' });
    }
  };

  const formatDate = (iso: string) => format(new Date(iso), 'MMM d, yyyy');

  const filteredTerms = terms.filter((t) => {
    const hay = `${t.academic_year} ${t.semester}`.toLowerCase();
    return hay.includes(searchTerm.toLowerCase());
  });

  return (
    <Layout>
      <div className="flex-1 space-y-4 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold tracking-tight">ALLOWED TERMS</h2>
            <p className="text-sm text-muted-foreground">Add and manage allowed academic terms</p>
          </div>
          <Button 
            className="bg-gradient-primary shadow-glow h-9"
            onClick={() => setIsFormOpen(true)}
          >
            <Plus className="mr-2 h-4 w-4" /> Add Term
          </Button>
        </div>

        {/* Add Term Modal */}
        <Dialog open={isFormOpen} onOpenChange={setIsFormOpen}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>New Allowed Term</DialogTitle>
              <DialogDescription>Provide academic year, semester, and date range.</DialogDescription>
            </DialogHeader>
            <div className="grid gap-4">
              <div className="flex flex-col gap-2">
                <Label htmlFor="ay">Academic Year</Label>
                <Input
                  id="ay"
                  placeholder="e.g., 2024-2025"
                  value={formData.academic_year}
                  onChange={(e) => setFormData((p) => ({ ...p, academic_year: e.target.value }))}
                />
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="sem">Semester</Label>
                <Input
                  id="sem"
                  placeholder="e.g., 1st Semester"
                  value={formData.semester}
                  onChange={(e) => setFormData((p) => ({ ...p, semester: e.target.value }))}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="flex flex-col gap-2">
                  <Label htmlFor="start">Start Date</Label>
                  <Input
                    id="start"
                    type="date"
                    value={formData.start_date}
                    onChange={(e) => setFormData((p) => ({ ...p, start_date: e.target.value }))}
                  />
                </div>
                <div className="flex flex-col gap-2">
                  <Label htmlFor="end">End Date</Label>
                  <Input
                    id="end"
                    type="date"
                    value={formData.end_date}
                    onChange={(e) => setFormData((p) => ({ ...p, end_date: e.target.value }))}
                  />
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsFormOpen(false)}>Cancel</Button>
              <Button 
                onClick={handleCreate}
                disabled={!formData.academic_year || !formData.semester || !formData.start_date || !formData.end_date}
              >
                Save
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Search */}
        <div className="flex items-center space-x-4">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search allowed terms..."
              className="pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>

        {/* Terms List */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <CalendarDays className="h-5 w-5" />
                  Allowed Terms
                </CardTitle>
                <CardDescription>All defined academic year + semester terms</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center h-32">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary"></div>
              </div>
            ) : filteredTerms.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Academic Year</TableHead>
                    <TableHead>Semester</TableHead>
                    <TableHead>Start</TableHead>
                    <TableHead>End</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredTerms.map((t) => (
                    <TableRow key={t.id}>
                      <TableCell className="font-medium">{t.academic_year}</TableCell>
                      <TableCell>{t.semester}</TableCell>
                      <TableCell>{formatDate(t.start_date)}</TableCell>
                      <TableCell>{formatDate(t.end_date)}</TableCell>
                      <TableCell className="text-right">
                        <Button variant="ghost" size="icon" onClick={() => handleDelete(t.id)}>
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <div className="text-center py-12">
                <Calendar className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium">No allowed terms found</h3>
                <p className="text-sm text-muted-foreground">Add your first allowed term using the form above.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
};

export default AllowedTerms;
