import * as XLSX from 'xlsx';

// Define the headers based on the StudentForm fields
const headers = [
  'surname',
  'middle_initial',
  'firstname',
  'student_id',
  'program',
  'year',
  'section',
  'sex'
];

// Create a new workbook
const wb = XLSX.utils.book_new();

// Add a worksheet with the headers
const ws = XLSX.utils.aoa_to_sheet([headers]);

// Add some example data
const exampleData = [
  {
    surname: 'Doe',
    middle_initial: 'M',
    firstname: 'John',
    student_id: '2023-001',
    program: 'Computer Science',
    year: '1st',
    section: 'BSCS 1A',
    sex: 'Male'
  },
  {
    surname: 'Smith',
    middle_initial: 'A',
    firstname: 'Jane',
    student_id: '2023-002',
    program: 'Information Technology',
    year: '2nd',
    section: 'BSIT 2B',
    sex: 'Female'
  }
];

// Add the example data to the worksheet
XLSX.utils.sheet_add_json(ws, exampleData, { header: headers, skipHeader: true, origin: 'A2' });

// Add the worksheet to the workbook
XLSX.utils.book_append_sheet(wb, ws, 'Students');

// Generate the Excel file
const excelBuffer = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });

// Create a Blob and download link
const blob = new Blob([excelBuffer], { type: 'application/octet-stream' });
const url = URL.createObjectURL(blob);
const link = document.createElement('a');
link.href = url;
link.download = 'student_import_template.xlsx';
document.body.appendChild(link);
link.click();
document.body.removeChild(link);
URL.revokeObjectURL(url);
