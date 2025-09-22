import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Search, Plus, Edit, Trash2, ChevronsUp, ChevronsDown, Loader2, List } from 'lucide-react';
import Layout from '@/components/Layout';
import PageWrapper from '@/components/PageWrapper';
import { supabase } from '@/lib/supabase';
import { toast } from 'sonner';

type AccountStatus = 'active' | 'inactive' | 'pending' | 'suspended';

interface OfficerAccount {
  id: string;
  email: string;
  first_name: string | null;
  last_name: string | null;
  role: string;
  status: AccountStatus;
  created_at: string;
}

// Role Component
const RoleDisplay = ({ role }: { role: string }) => {
  return (
    <span className="text-xs text-gray-500">
      {role.charAt(0).toUpperCase() + role.slice(1)}
    </span>
  );
};

// Status Component
const StatusDisplay = ({ status }: { status: AccountStatus }) => {
  return (
    <span className="text-xs text-gray-500 capitalize">
      {status}
    </span>
  );
};


export default function OfficerAccounts() {
  const [accounts, setAccounts] = useState<OfficerAccount[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRole, setSelectedRole] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [displayPageSize, setDisplayPageSize] = useState(10);
  const [isLoading, setIsLoading] = useState(true);
  
  // Sorting
  type AccountSortKey = 'user' | 'email' | 'role' | 'status' | 'created';
  const [sortKey, setSortKey] = useState<AccountSortKey>('user');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  // Filter accounts
  const filteredAccounts = accounts.filter(account => {
    const matchesSearch = 
      `${account.first_name} ${account.last_name}`.toLowerCase().includes(searchTerm.toLowerCase()) ||
      account.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRole = selectedRole === 'all' || account.role === selectedRole;
    const matchesStatus = selectedStatus === 'all' || account.status === selectedStatus;
    
    return matchesSearch && matchesRole && matchesStatus;
  });

  const sortedAccounts = [...filteredAccounts].sort((a, b) => {
    const dir = sortDir === 'asc' ? 1 : -1;
    const nameA = `${a.first_name || ''} ${a.last_name || ''}`.trim().toLowerCase();
    const nameB = `${b.first_name || ''} ${b.last_name || ''}`.trim().toLowerCase();
    switch (sortKey) {
      case 'user':
        return nameA.localeCompare(nameB) * dir || a.email.localeCompare(b.email) * dir;
      case 'email':
        return a.email.localeCompare(b.email) * dir;
      case 'role':
        return (a.role || '').toString().localeCompare((b.role || '').toString()) * dir;
      case 'status':
        return (a.status || '').toString().localeCompare((b.status || '').toString()) * dir;
      case 'created':
        return (new Date(a.created_at).getTime() - new Date(b.created_at).getTime()) * dir;
      default:
        return 0;
    }
  });

  const handleSort = (key: AccountSortKey) => {
    if (sortKey === key) setSortDir(prev => (prev === 'asc' ? 'desc' : 'asc'));
    else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const handlePageSizeChange = (size: number) => {
    setDisplayPageSize(size);
  };

  const handleEdit = (id: string) => {
    console.log('Edit account:', id);
    // TODO: Implement edit functionality
  };

  const handleDelete = (id: string) => {
    console.log('Delete account:', id);
    // TODO: Implement delete functionality
  };

  // Load ROTC Officer accounts from database
  const loadOfficerAccounts = async () => {
    setIsLoading(true);
    try {
      const { data, error } = await supabase
        .from('users')
        .select('*')
        .eq('role', 'ROTC officer')
        .order('created_at', { ascending: false });

      if (error) throw error;
      
      setAccounts(data || []);
    } catch (error) {
      console.error('Error loading officer accounts:', error);
      toast.error('Failed to load officer accounts');
    } finally {
      setIsLoading(false);
    }
  };

  // Load accounts on component mount
  useEffect(() => {
    loadOfficerAccounts();
  }, []);

  const handleAddAccount = () => {
    console.log('Add new account');
    // TODO: Implement add account functionality
  };

  if (isLoading) {
    return (
      <Layout>
        <div className="container mx-auto p-4">
          <div className="text-center flex items-center justify-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading officer accounts...
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <div className="px-6 py-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-lg font-bold tracking-tight">OFFICER ACCOUNT MANAGEMENT</h1>
            </div>
          </div>
          
          {/* Match Allowed Terms spacing between title and card */}
          <div className="mb-16"></div>

          <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-base font-semibold text-education-navy">List of Officer Accounts</h3>
              <Button onClick={handleAddAccount} className="flex items-center gap-2 h-8 text-xs">
                <Plus className="h-3 w-3" />
                Add Account
              </Button>
            </div>
            
            {/* Big space below List of Accounts label */}
            <div className="mb-8"></div>
            
            {/* Show search and filters inside the card */}
            <div className="flex items-center justify-between gap-4 p-0 mb-3">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-600">Showed:</span>
                <Select
                  value={displayPageSize >= 999999 ? "all" : displayPageSize.toString()}
                  onValueChange={(value) => {
                    if (value === "all") {
                      handlePageSizeChange(999999);
                    } else {
                      handlePageSizeChange(parseInt(value));
                    }
                  }}
                >
                  <SelectTrigger className="h-8 w-24">
                    <SelectValue>
                      {displayPageSize.toString()}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="10">10</SelectItem>
                    <SelectItem value="100">100</SelectItem>
                    <SelectItem value="250">250</SelectItem>
                    <SelectItem value="all">ALL</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-600">Search:</span>
                <div className="relative min-w-[240px] max-w-[340px]">
                  <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                  <Input
                    type="search"
                    placeholder="Search accounts..."
                    className="pl-7 pr-7 h-8 w-full text-sm bg-background border-border focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all duration-200"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                </div>
              </div>
            </div>
            
            <div className="border-t border-gray-200 overflow-hidden min-h-[378px]">
              <table className="min-w-full divide-y divide-gray-200 border-b border-gray-200">
                <thead className="bg-gray-50">
                  <tr className="text-xs text-black h-8">
                    <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                      <div className="flex items-center gap-1">
                        User
                        <button type="button" onClick={() => handleSort('user')} className="p-0.5 text-gray-500 hover:text-black">
                          {sortKey === 'user' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5" /> : <ChevronsDown className="w-3.5 h-3.5" />) : <ChevronsUp className="w-3.5 h-3.5 opacity-40" />}
                        </button>
                      </div>
                    </th>
                    <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                      <div className="flex items-center gap-1">
                        Email
                        <button type="button" onClick={() => handleSort('email')} className="p-0.5 text-gray-500 hover:text-black">
                          {sortKey === 'email' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5" /> : <ChevronsDown className="w-3.5 h-3.5" />) : <ChevronsUp className="w-3.5 h-3.5 opacity-40" />}
                        </button>
                      </div>
                    </th>
                    <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                      <div className="flex items-center gap-1">
                        Role
                        <button type="button" onClick={() => handleSort('role')} className="p-0.5 text-gray-500 hover:text-black">
                          {sortKey === 'role' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5" /> : <ChevronsDown className="w-3.5 h-3.5" />) : <ChevronsUp className="w-3.5 h-3.5 opacity-40" />}
                        </button>
                      </div>
                    </th>
                    <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                      <div className="flex items-center gap-1">
                        Status
                        <button type="button" onClick={() => handleSort('status')} className="p-0.5 text-gray-500 hover:text-black">
                          {sortKey === 'status' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5" /> : <ChevronsDown className="w-3.5 h-3.5" />) : <ChevronsUp className="w-3.5 h-3.5 opacity-40" />}
                        </button>
                      </div>
                    </th>
                    <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                      <div className="flex items-center gap-1">
                        Created
                        <button type="button" onClick={() => handleSort('created')} className="p-0.5 text-gray-500 hover:text-black">
                          {sortKey === 'created' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5" /> : <ChevronsDown className="w-3.5 h-3.5" />) : <ChevronsUp className="w-3.5 h-3.5 opacity-40" />}
                        </button>
                      </div>
                    </th>
                    <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">Actions</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200 text-xs text-gray-500">
                  {filteredAccounts.length === 0 ? (
                    <tr className="h-8">
                      <td colSpan={6} className="px-3 py-1 text-center text-sm text-gray-500">
                        No accounts found matching the current filters.
                      </td>
                    </tr>
                  ) : (
                    <>
                      {sortedAccounts.slice(0, displayPageSize).map((account) => (
                        <tr key={account.id} className="hover:bg-gray-50 h-8">
                          <td className="px-3 py-1 whitespace-nowrap">
                            <div className="font-medium">
                              {account.first_name && account.last_name 
                                ? `${account.first_name} ${account.last_name}`
                                : account.email.split('@')[0]}
                            </div>
                          </td>
                          <td className="px-3 py-1 whitespace-nowrap">
                            {account.email}
                          </td>
                          <td className="px-3 py-1 whitespace-nowrap">
                            <RoleDisplay role={account.role} />
                          </td>
                          <td className="px-3 py-1 whitespace-nowrap">
                            <StatusDisplay status={account.status} />
                          </td>
                          <td className="px-3 py-1 whitespace-nowrap">
                            {new Date(account.created_at).toLocaleDateString()}
                          </td>
                          <td className="px-3 py-1 whitespace-nowrap">
                            <div className="flex items-center gap-1">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleEdit(account.id)}
                                className="h-6 w-6 p-0 text-xs"
                              >
                                <Edit className="h-3 w-3 text-yellow-600 transform hover:scale-125 transition-transform duration-200 ease-in-out" />
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleDelete(account.id)}
                                className="h-6 w-6 p-0 text-xs"
                              >
                                <Trash2 className="h-3 w-3 text-red-600 transform hover:scale-125 transition-transform duration-200 ease-in-out" />
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </PageWrapper>
    </Layout>
  );
}
