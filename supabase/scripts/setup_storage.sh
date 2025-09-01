#!/bin/bash

# This script sets up the necessary storage buckets in Supabase
# It should be run after the initial database setup

# Exit on error
set -e

# Check if SUPABASE_URL and SUPABASE_SERVICE_KEY are set
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_SERVICE_KEY" ]; then
  echo "Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"
  exit 1
fi

# Function to create a storage bucket
create_bucket() {
  local bucket_name=$1
  local is_public=$2
  
  echo "Creating bucket: $bucket_name"
  
  # Create the bucket
  curl -X POST "$SUPABASE_URL/storage/v1/bucket" \
    -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$bucket_name\",\"public\":$is_public}"
  
  echo -e "\nBucket $bucket_name created\n"
}

# Create buckets
create_bucket "signatures" true

# Update CORS policy for the signatures bucket
curl -X PUT "$SUPABASE_URL/storage/v1/bucket/signatures/cors" \
  -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "allowedOrigins": [
      "http://localhost:3000",
      "http://localhost:5173",
      "https://your-production-domain.com"
    ],
    "allowedMethods": ["GET", "POST", "PUT", "DELETE"],
    "allowedHeaders": ["*"],
    "exposeHeaders": ["Content-Length", "ETag"],
    "maxAgeSeconds": 3600
  }'

echo "Storage setup complete!"
