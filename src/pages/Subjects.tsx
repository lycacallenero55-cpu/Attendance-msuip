import Layout from "@/components/Layout";
import PageWrapper from "@/components/PageWrapper";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Plus, Eye, Edit, Trash2, ChevronsUp, ChevronsDown, List } from "lucide-react";
import { useState, useMemo } from "react";

type Subject = {
  id: number;
  subject_code: string;
  subject_title: string;
  enrolled_students: number;
  instructor: string;
};

const Subjects = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [displayPageSize, setDisplayPageSize] = useState(10);
  
  // Sorting state
  type SubjectSortKey = 'code' | 'title' | 'students' | 'instructor';
  const [sortKey, setSortKey] = useState<SubjectSortKey>('code');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  // Mock data
  const mockSubjects: Subject[] = [
    { id: 1, subject_code: 'CS 101', subject_title: 'Introduction to Computer Science', enrolled_students: 45, instructor: 'Dr. Smith' },
    { id: 2, subject_code: 'MATH 201', subject_title: 'Calculus II', enrolled_students: 32, instructor: 'Prof. Johnson' },
    { id: 3, subject_code: 'ENG 102', subject_title: 'English Literature', enrolled_students: 28, instructor: 'Dr. Williams' },
    { id: 4, subject_code: 'PHYS 301', subject_title: 'Quantum Physics', enrolled_students: 15, instructor: 'Prof. Brown' },
    { id: 5, subject_code: 'CHEM 101', subject_title: 'General Chemistry', enrolled_students: 38, instructor: 'Dr. Davis' },
    { id: 6, subject_code: 'HIST 202', subject_title: 'World History', enrolled_students: 42, instructor: 'Prof. Miller' },
    { id: 7, subject_code: 'ART 150', subject_title: 'Digital Art Fundamentals', enrolled_students: 22, instructor: 'Dr. Wilson' },
    { id: 8, subject_code: 'BIO 201', subject_title: 'Molecular Biology', enrolled_students: 19, instructor: 'Prof. Moore' },
    { id: 9, subject_code: 'ECON 101', subject_title: 'Microeconomics', enrolled_students: 35, instructor: 'Dr. Taylor' },
    { id: 10, subject_code: 'PSYC 301', subject_title: 'Cognitive Psychology', enrolled_students: 26, instructor: 'Prof. Anderson' },
    { id: 11, subject_code: 'CS 301', subject_title: 'Data Structures and Algorithms', enrolled_students: 41, instructor: 'Dr. Thomas' },
    { id: 12, subject_code: 'MATH 301', subject_title: 'Linear Algebra', enrolled_students: 29, instructor: 'Prof. Jackson' },
  ];

  const [subjects] = useState<Subject[]>(mockSubjects);

  // Filter and sort subjects
  const filteredSubjects = useMemo(() => {
    const filtered = subjects.filter(subject =>
      subject.subject_code.toLowerCase().includes(searchTerm.toLowerCase()) ||
      subject.subject_title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      subject.instructor.toLowerCase().includes(searchTerm.toLowerCase())
    );

    // Sort
    filtered.sort((a, b) => {
      let aValue: string | number, bValue: string | number;
      
      switch (sortKey) {
        case 'code':
          aValue = a.subject_code;
          bValue = b.subject_code;
          break;
        case 'title':
          aValue = a.subject_title;
          bValue = b.subject_title;
          break;
        case 'students':
          aValue = a.enrolled_students;
          bValue = b.enrolled_students;
          break;
        case 'instructor':
          aValue = a.instructor;
          bValue = b.instructor;
          break;
        default:
          aValue = a.subject_code;
          bValue = b.subject_code;
      }

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortDir === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      } else {
        return sortDir === 'asc' 
          ? aValue - bValue
          : bValue - aValue;
      }
    });

    return filtered;
  }, [subjects, searchTerm, sortKey, sortDir]);

  const handleSort = (key: SubjectSortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const handlePageSizeChange = (size: number) => {
    setDisplayPageSize(size);
  };

  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <div className="px-6 py-4">
        <div className="mb-3">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-1">
            <div>
              <h1 className="text-2xl font-bold text-education-navy">SUBJECTS</h1>
            </div>
          </div>
        </div>
        
        {/* Big space between page title and card */}
        <div className="mb-16"></div>
        
        {/* Search and Subjects Section */}
        <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-base font-semibold text-education-navy">List of Subjects</h3>
            <div className="mt-0">
              <Button
                size="sm"
                className="h-8"
                onClick={() => {
                  // TODO: Implement add subject functionality
                }}
              >
                <Plus className="w-4 h-4 mr-1" />
                Add Subject
              </Button>
            </div>
          </div>
          
          {/* Big space below List of Subjects label */}
          <div className="mb-8"></div>
          
          {/* Top controls row */}
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
                  placeholder="Search subjects..."
                  className="pl-7 pr-7 h-8 w-full text-sm bg-background border-border focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all duration-200 [&::-webkit-search-cancel-button]:hidden"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  type="search"
                />
              </div>
            </div>
          </div>

          {/* Table View */}
          <div className="border-t border-gray-200 overflow-hidden min-h-[378px]">
                <table className="min-w-full divide-y divide-gray-200 border-b border-gray-200">
              <thead className="bg-gray-50">
                <tr className="text-xs text-black h-8">
                  <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                    <div className="flex items-center gap-1">Code
                      <button type="button" onClick={() => handleSort('code')} className="p-0.5 text-gray-500 hover:text-black">
                        {sortKey === 'code' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5 text-black"/> : <ChevronsDown className="w-3.5 h-3.5 text-black"/>) : <ChevronsUp className="w-3.5 h-3.5 opacity-40 text-black"/>}
                      </button>
                    </div>
                  </th>
                  <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                    <div className="flex items-center gap-1">Descriptive Title
                      <button type="button" onClick={() => handleSort('title')} className="p-0.5 text-gray-500 hover:text-black">
                        {sortKey === 'title' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5"/> : <ChevronsDown className="w-3.5 h-3.5"/>) : <ChevronsUp className="w-3.5 h-3.5 opacity-40"/>}
                      </button>
                    </div>
                  </th>
                  <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                    <div className="flex items-center gap-1">Students
                      <button type="button" onClick={() => handleSort('students')} className="p-0.5 text-gray-500 hover:text-black">
                        {sortKey === 'students' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5"/> : <ChevronsDown className="w-3.5 h-3.5"/>) : <ChevronsUp className="w-3.5 h-3.5 opacity-40"/>}
                      </button>
                    </div>
                  </th>
                  <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                    <div className="flex items-center gap-1">Instructor
                      <button type="button" onClick={() => handleSort('instructor')} className="p-0.5 text-gray-500 hover:text-black">
                        {sortKey === 'instructor' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5"/> : <ChevronsDown className="w-3.5 h-3.5"/>) : <ChevronsUp className="w-3.5 h-3.5 opacity-40"/>}
                      </button>
                    </div>
                  </th>
                  <th scope="col" className="px-3 py-2 text-left font-semibold uppercase"></th> {/* Empty for actions */}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200 text-xs text-gray-500">
                {filteredSubjects.length === 0 ? (
                  <tr className="h-8">
                    <td colSpan={5} className="px-3 py-1 text-center text-sm text-gray-500">
                      {subjects.length === 0 
                        ? 'No subjects found. Add your first subject!'
                        : 'No subjects match the current search. Try adjusting your search.'}
                    </td>
                  </tr>
                ) : (
                  <>
                    {filteredSubjects.slice(0, displayPageSize >= 999999 ? filteredSubjects.length : displayPageSize).map((subject) => (
                      <tr key={subject.id} className="hover:bg-gray-50 h-8">
                        <td className="px-3 py-1 whitespace-nowrap">
                          <div className="font-medium">{subject.subject_code}</div>
                        </td>
                        <td className="px-3 py-1 whitespace-nowrap">
                          <span className="truncate max-w-[200px] inline-block">{subject.subject_title}</span>
                        </td>
                        <td className="px-3 py-1 whitespace-nowrap">
                          {subject.enrolled_students}
                        </td>
                        <td className="px-3 py-1 whitespace-nowrap">
                          {subject.instructor}
                        </td>
                        <td className="px-3 py-1 whitespace-nowrap text-right">
                          <div className="flex gap-1 justify-end">
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-6 w-6 p-0"
                              onClick={() => {
                                // TODO: Implement view functionality
                              }}
                            >
                              <List className="h-3 w-3 text-green-600" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-6 w-6 p-0"
                              onClick={() => {
                                // TODO: Implement edit functionality
                              }}
                            >
                              <Edit className="h-3 w-3 text-yellow-600" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-6 w-6 p-0"
                              onClick={() => {
                                // TODO: Implement delete functionality
                              }}
                            >
                              <Trash2 className="h-3 w-3 text-red-600" />
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
};

export default Subjects;