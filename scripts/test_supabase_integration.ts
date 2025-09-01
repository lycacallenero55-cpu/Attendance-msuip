import { createClient } from '@supabase/supabase-js';
import 'dotenv/config';

// Load environment variables
const supabaseUrl = process.env.VITE_SUPABASE_URL;
const supabaseKey = process.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('Missing Supabase environment variables');
  process.exit(1);
}

// Initialize Supabase client
const supabase = createClient(supabaseUrl, supabaseKey);

// Test data
const testUser = {
  email: 'test@example.com',
  password: 'testpassword123',
  name: 'Test User',
};

const testSession = {
  title: 'Test Session',
  type: 'class',
  date: new Date().toISOString().split('T')[0],
  time: '10:00',
  location: 'Test Location',
  instructor: 'Test Instructor',
  capacity: 30,
  program: 'CS',
  year: '2023',
  section: 'A',
  description: 'Test session description',
};

const testStudent = {
  student_id: 'S12345',
  firstname: 'John',
  surname: 'Doe',
  program: 'CS',
  year: '2023',
  section: 'A',
};

async function runTests() {
  console.log('Starting Supabase integration tests...\n');

  try {
    // Test 1: Sign up a new user
    console.log('Test 1: User sign up');
    const { data: signUpData, error: signUpError } = await supabase.auth.signUp({
      email: testUser.email,
      password: testUser.password,
      options: {
        data: {
          full_name: testUser.name,
        },
      },
    });

    if (signUpError) throw signUpError;
    console.log('✅ User signed up successfully');

    // Test 2: Sign in
    console.log('\nTest 2: User sign in');
    const { data: signInData, error: signInError } = await supabase.auth.signInWithPassword({
      email: testUser.email,
      password: testUser.password,
    });

    if (signInError) throw signInError;
    console.log('✅ User signed in successfully');

    const userId = signInData.user?.id;
    if (!userId) throw new Error('No user ID returned after sign in');

    // Test 3: Create a session (requires instructor role)
    console.log('\nTest 3: Create a session');
    // First, update user role to instructor
    const { error: roleError } = await supabase
      .from('profiles')
      .update({ role: 'instructor' })
      .eq('id', userId);

    if (roleError) throw roleError;
    
    const { data: sessionData, error: sessionError } = await supabase
      .from('sessions')
      .insert(testSession)
      .select()
      .single();

    if (sessionError) throw sessionError;
    console.log('✅ Session created successfully:', sessionData.id);

    // Test 4: Create a student
    console.log('\nTest 4: Create a student');
    const { data: studentData, error: studentError } = await supabase
      .from('students')
      .insert(testStudent)
      .select()
      .single();

    if (studentError) throw studentError;
    console.log('✅ Student created successfully:', studentData.id);

    // Test 5: Mark attendance
    console.log('\nTest 5: Mark attendance');
    const { data: attendanceData, error: attendanceError } = await supabase
      .from('attendance')
      .insert({
        session_id: sessionData.id,
        student_id: studentData.id,
        status: 'present',
        time_in: new Date().toISOString(),
      })
      .select()
      .single();

    if (attendanceError) throw attendanceError;
    console.log('✅ Attendance marked successfully:', attendanceData.id);

    // Test 6: Query attendance
    console.log('\nTest 6: Query attendance');
    const { data: attendanceQuery, error: queryError } = await supabase
      .from('attendance')
      .select(`
        *,
        students(*),
        sessions(*)
      `)
      .eq('id', attendanceData.id)
      .single();

    if (queryError) throw queryError;
    console.log('✅ Attendance queried successfully');
    console.log('   Student:', attendanceQuery.students?.firstname, attendanceQuery.students?.surname);
    console.log('   Session:', attendanceQuery.sessions?.title);
    console.log('   Status:', attendanceQuery.status);

    // Test 7: Clean up (delete test data)
    console.log('\nTest 7: Clean up test data');
    await supabase.from('attendance').delete().eq('id', attendanceData.id);
    await supabase.from('students').delete().eq('id', studentData.id);
    await supabase.from('sessions').delete().eq('id', sessionData.id);
    await supabase.auth.admin.deleteUser(userId);
    
    console.log('✅ Test data cleaned up');

    console.log('\n🎉 All tests passed successfully!');
  } catch (error) {
    console.error('\n❌ Test failed:', error);
    process.exit(1);
  }
}

runTests();
