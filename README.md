# Attendance Monitoring System

A comprehensive attendance tracking system built with React, TypeScript, and Supabase, featuring a fully serverless architecture.

## Features

- **User Authentication**: Secure login with email/password using Supabase Auth
- **Student Management**: Add, edit, and manage student records
- **Session Management**: Create and manage class/event sessions
- **Attendance Tracking**: Mark and track student attendance
- **Digital Signatures**: Capture and store student signatures
- **Reports & Analytics**: Generate attendance reports and statistics

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Shadcn UI
- **Backend Services**: Supabase (PostgreSQL, Auth, Storage, Edge Functions)
- **Database**: Supabase PostgreSQL with Row Level Security (RLS)
- **Authentication**: Supabase Auth with email/password and social providers
- **File Storage**: Supabase Storage for digital signatures
- **Deployment**: Vercel, Netlify, or any static hosting platform

## Prerequisites

- Node.js 18+ and npm/yarn
- Supabase account and project
- Git for version control

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/attendance-monitoring-system.git
cd attendance-monitoring-system
```

### 2. Set up environment variables

1. Copy the example environment file:

```bash
cp attendance-frontend/.env.example attendance-frontend/.env
```

2. Edit the `.env` file with your Supabase project details:

```env
# Supabase
VITE_SUPABASE_URL=your-supabase-project-url
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
```

### 3. Set up Supabase

1. Create a new project in the [Supabase Dashboard](https://app.supabase.com/)
2. Run the database migrations:
   ```bash
   # In the supabase/migrations directory, run the SQL files in order:
   # 001_initial_schema.sql
   # 002_rls_policies.sql
   ```
3. Set up storage buckets:
   ```bash
   # Set your Supabase URL and service key as environment variables
   export SUPABASE_URL=your-supabase-project-url
   export SUPABASE_SERVICE_KEY=your-supabase-service-key
   
   # Run the storage setup script
   bash supabase/scripts/setup_storage.sh
   ```

### 4. Install dependencies

```bash
cd attendance-frontend
npm install
```

### 5. Run the development server

```bash
npm run dev
```

Your application should now be running at `http://localhost:5173`

## Testing

To run the Supabase integration tests:

1. Install the required dependencies:
   ```bash
   npm install -D typescript ts-node dotenv @supabase/supabase-js
   ```

2. Run the test script:
   ```bash
   cd scripts
   npx ts-node test_supabase_integration.ts
   ```

## Deployment

### Vercel (Recommended)

1. Push your code to a GitHub/GitLab/Bitbucket repository
2. Import the project to Vercel
3. Add the environment variables from your `.env` file
4. Deploy!

### Netlify

1. Push your code to a GitHub/GitLab/Bitbucket repository
2. Create a new site from Git in Netlify
3. Set the build command: `npm run build`
4. Set the publish directory: `dist`
5. Add the environment variables
6. Deploy!

## Project Structure

```
attendance-monitoring-system/
├── attendance-frontend/     # Frontend React application
│   ├── public/             # Static files
│   ├── src/                # Source code
│   │   ├── components/     # Reusable UI components
│   │   ├── contexts/       # React contexts
│   │   ├── hooks/          # Custom React hooks
│   │   ├── lib/            # Utility functions and services
│   │   ├── pages/          # Page components
│   │   ├── types/          # TypeScript type definitions
│   │   └── App.tsx         # Main application component
│   └── ...
├── supabase/               # Supabase configuration
│   ├── migrations/         # Database migrations
│   └── scripts/            # Utility scripts
└── ...
```

## Security

- All database tables have Row Level Security (RLS) enabled
- File uploads are restricted to authenticated users
- User roles (admin, instructor, user) control access to different features
- Sensitive operations require proper authentication and authorization

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_JWT_SECRET=your_supabase_jwt_secret
BUCKET_NAME=signatures

# Frontend
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 3. Run the application

#### Development Mode

```bash
# Make the deployment script executable
chmod +x deploy.sh

# Start the development environment
./deploy.sh dev
```

This will start:
- Backend server at http://localhost:5000
- Frontend development server at http://localhost:3000

#### Production Mode

```bash
# Make the deployment script executable
chmod +x deploy.sh

# Start the production environment
./deploy.sh prod
```

This will build and start the production containers in detached mode.

## Project Structure

```
attendance-monitoring-system/
├── attendance-backend/       # FastAPI backend
│   ├── alembic/              # Database migrations
│   ├── app/                  # Application code
│   │   ├── api/              # API routes
│   │   ├── core/             # Core functionality
│   │   ├── db/               # Database models and session
│   │   ├── services/         # Business logic
│   │   └── main.py           # FastAPI application
│   ├── tests/                # Backend tests
│   ├── .env.example          # Example environment variables
│   ├── Dockerfile            # Backend Dockerfile
│   ├── pyproject.toml        # Python dependencies
│   └── README.md             # Backend documentation
│
├── attendance-frontend/      # React frontend
│   ├── public/               # Static files
│   ├── src/                  # Source code
│   │   ├── components/       # Reusable components
│   │   ├── pages/            # Page components
│   │   ├── services/         # API services
│   │   ├── styles/           # Global styles
│   │   └── main.tsx          # Application entry point
│   ├── .env.example          # Example environment variables
│   ├── Dockerfile            # Frontend Dockerfile
│   ├── package.json          # Node.js dependencies
│   └── README.md             # Frontend documentation
│
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile                # Main Dockerfile
├── .env.example              # Example environment variables
├── deploy.sh                 # Deployment script
└── README.md                 # This file
```

## Development

### Backend Development

1. Navigate to the backend directory:
   ```bash
   cd attendance-backend
   ```

2. Install Python dependencies:
   ```bash
   poetry install
   ```

3. Run the development server:
   ```bash
   poetry run uvicorn app.main:app --reload
   ```

### Frontend Development

1. Navigate to the frontend directory:
   ```bash
   cd attendance-frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## Deployment

### Prerequisites

- Docker and Docker Compose installed on the server
- Domain name (for production)
- SSL certificates (recommended for production)

### Production Deployment

1. Copy the environment file and update it with your production values:
   ```bash
   cp .env.example .env
   # Edit .env with your production values
   ```

2. Make the deployment script executable:
   ```bash
   chmod +x deploy.sh
   ```

3. Deploy the application:
   ```bash
   ./deploy.sh prod
   ```

4. Set up a reverse proxy (Nginx/Apache) to handle SSL termination and route traffic to the application.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [Supabase](https://supabase.com/)
- [Shadcn UI](https://ui.shadcn.com/)
