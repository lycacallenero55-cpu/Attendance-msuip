import { useState, useEffect, useRef } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { toast } from 'sonner';
import { uploadSignature, getStudentSignatures, getPrimarySignature } from '@/lib/supabaseService';
import type { Signature, SignatureUploadOptions } from '@/types';
import { Download } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Info, AlertCircle, CheckCircle2, Upload, X } from 'lucide-react';
import { Progress } from '@/components/ui/progress';

// Constants
const MIN_SIGNATURES = 5;
const MAX_FILE_SIZE_MB = 2;
const ALLOWED_FILE_TYPES = ['image/png', 'image/jpeg', 'image/webp'];

// Using the Signature type from @/types instead of local interface

interface StudentSignaturesProps {
  studentId: number;
  studentName: string;
}

export function StudentSignatures({ studentId, studentName }: StudentSignaturesProps) {
  const [signatures, setSignatures] = useState<Signature[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const signatureCount = signatures.length;
  const isComplete = signatureCount >= MIN_SIGNATURES;
  const primarySignature = signatures.find(sig => sig.id === signatures[0]?.id); // Simplified for now

  // Load signatures when component mounts or studentId changes
  useEffect(() => {
    const loadSignatures = async () => {
      try {
        setIsLoading(true);
        const sigs = await getStudentSignatures(studentId);
        setSignatures(sigs);
      } catch (error) {
        console.error('Error loading signatures:', error);
        toast.error('Failed to load signatures');
      } finally {
        setIsLoading(false);
      }
    };

    if (studentId) {
      loadSignatures();
    }
  }, [studentId]);
  
  // Clean up preview URL when component unmounts
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const validateFile = (file: File): { isValid: boolean; error?: string } => {
    // Check file type
    if (!ALLOWED_FILE_TYPES.includes(file.type)) {
      return { 
        isValid: false, 
        error: `Invalid file type. Please upload a PNG, JPEG, or WebP image.` 
      };
    }

    // Check file size (2MB max)
    const maxSize = MAX_FILE_SIZE_MB * 1024 * 1024;
    if (file.size > maxSize) {
      return { 
        isValid: false, 
        error: `File is too large. Maximum size is ${MAX_FILE_SIZE_MB}MB.` 
      };
    }

    return { isValid: true };
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    const validation = validateFile(file);
    if (!validation.isValid) {
      setValidationError(validation.error || 'Invalid file');
      return;
    }

    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    setPreviewUrl(previewUrl);
    setSelectedFile(file);
    setValidationError(null);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setValidationError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || !studentId) return;

    try {
      setIsUploading(true);
      setUploadProgress(0);

      // Create metadata for the signature
      const options: SignatureUploadOptions = {
        deviceInfo: {
          userAgent: navigator.userAgent,
          screenResolution: `${window.screen.width}x${window.screen.height}`,
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        },
        fileInfo: {
          size: selectedFile.size,
          type: selectedFile.type,
        },
      };

      // Upload the signature with progress tracking
      const { signature: newSignature } = await uploadSignature(
        studentId,
        selectedFile,
        {
          deviceInfo: options.deviceInfo,
          fileInfo: options.fileInfo,
          // Add any additional metadata here
        },
        (progress: number) => {
          setUploadProgress(progress);
        }
      );

      // Add the new signature to the list
      setSignatures(prev => [newSignature, ...prev]);
      
      toast.success('Signature uploaded successfully');
      setSelectedFile(null);
      setPreviewUrl(null);
    } catch (error) {
      console.error('Error uploading signature:', error);
      toast.error('Failed to upload signature');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  // Format upload date from signature
  const formatUploadDate = (signature: Signature) => {
    return new Date(signature.created_at).toLocaleDateString();
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <div className="flex flex-col space-y-2">
          <div className="flex justify-between items-start">
            <div>
              <CardTitle>Signature Collection</CardTitle>
              <CardDescription>
                {studentName} • {isComplete ? (
                  <span className="text-green-600">Complete ({signatureCount}/{MIN_SIGNATURES})</span>
                ) : (
                  <span className="text-amber-600">In Progress ({signatureCount}/{MIN_SIGNATURES})</span>
                )}
              </CardDescription>
            </div>
            <div className="flex space-x-2">
              <Button 
                variant={isComplete ? "default" : "outline"}
                disabled={isUploading || !selectedFile}
                onClick={handleUpload}
                className="gap-2"
              >
                {isUploading ? (
                  <>
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4" />
                    Upload Signature
                  </>
                )}
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept={ALLOWED_FILE_TYPES.join(',')}
                className="hidden"
                onChange={handleFileChange}
                disabled={isUploading}
              />
            </div>
          </div>
          
          {!isComplete && (
            <Alert className="bg-amber-50 border-amber-200">
              <Info className="h-4 w-4 text-amber-600" />
              <AlertTitle>More signatures needed</AlertTitle>
              <AlertDescription className="text-amber-800">
                Please provide at least {MIN_SIGNATURES - signatureCount} more signature 
                sample{MIN_SIGNATURES - signatureCount !== 1 ? 's' : ''} for better accuracy.
              </AlertDescription>
            </Alert>
          )}

          {isComplete && (
            <Alert className="bg-green-50 border-green-200">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertTitle>Signature collection complete</AlertTitle>
              <AlertDescription className="text-green-800">
                You've provided enough signatures for accurate verification. You can still add more samples to improve accuracy.
              </AlertDescription>
            </Alert>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* File Selection & Preview */}
        <div className="space-y-4">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
            {previewUrl ? (
              <div className="space-y-4">
                <div className="relative bg-gray-50 rounded-md p-4">
                  <img 
                    src={previewUrl} 
                    alt="Signature preview" 
                    className="max-h-40 mx-auto object-contain"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute top-2 right-2 h-8 w-8 rounded-full bg-white/80 hover:bg-white"
                    onClick={handleRemoveFile}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                <p className="text-sm text-gray-500">
                  {selectedFile?.name} • {(selectedFile?.size || 0) / 1024 > 1024
                    ? `${(selectedFile!.size / (1024 * 1024)).toFixed(1)} MB`
                    : `${Math.ceil((selectedFile?.size || 0) / 1024)} KB`}
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="mx-auto h-12 w-12 text-gray-400">
                  <Upload className="h-full w-full" />
                </div>
                <div className="text-sm text-gray-600">
                  <label
                    htmlFor="file-upload"
                    className="relative cursor-pointer rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none"
                  >
                    <span>Upload a file</span>
                    <input
                      id="file-upload"
                      name="file-upload"
                      type="file"
                      className="sr-only"
                      accept={ALLOWED_FILE_TYPES.join(',')}
                      onChange={handleFileChange}
                      disabled={isUploading}
                    />
                  </label>
                  <p className="pl-1">or drag and drop</p>
                </div>
                <p className="text-xs text-gray-500">
                  PNG, JPG, WebP up to {MAX_FILE_SIZE_MB}MB
                </p>
              </div>
            )}
          </div>

          {validationError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{validationError}</AlertDescription>
            </Alert>
          )}

          {uploadProgress > 0 && uploadProgress < 100 && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-gray-600">
                <span>Uploading...</span>
                <span>{Math.round(uploadProgress)}%</span>
              </div>
              <Progress value={uploadProgress} className="h-2" />
            </div>
          )}
        </div>

        {/* Signature Gallery */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-3">Collected Signatures ({signatures.length})</h3>
          
          {isLoading ? (
            <div className="flex justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
            </div>
          ) : signatures.length === 0 ? (
            <div className="text-center py-8 text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">
              No signatures collected yet. Upload your first signature to begin.
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
              {signatures.map((signature) => (
                <div 
                  key={signature.id} 
                  className={`relative group border rounded-md overflow-hidden ${
                    primarySignature?.id === signature.id ? 'ring-2 ring-primary' : ''
                  }`}
                >
                  <img
                    src={`${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/signatures/${signature.storage_path}`}
                    alt={`Signature ${signature.id}`}
                    className="w-full h-24 object-contain bg-gray-50"
                    onError={(e) => {
                      // Fallback to a placeholder if the image fails to load
                      const target = e.target as HTMLImageElement;
                      target.src = '/signature-placeholder.png';
                    }}
                  />
                  <div className="absolute inset-0 bg-black bg-opacity-50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2 p-2">
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      className="text-white hover:bg-white/20"
                      onClick={(e) => {
                        e.stopPropagation();
                        // Handle download
                        const link = document.createElement('a');
                        link.href = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/signatures/${signature.storage_path}`;
                        link.download = signature.file_name;
                        link.click();
                      }}
                    >
                      <Download className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Signature Guidelines */}
        <Alert className="bg-blue-50 border-blue-200">
          <Info className="h-4 w-4 text-blue-600" />
          <div>
            <AlertTitle className="text-blue-800">Signature Guidelines</AlertTitle>
            <ul className="list-disc pl-5 mt-1 text-sm text-blue-700 space-y-1">
              <li>Sign naturally, as you would on a document</li>
              <li>Use a dark pen on a light background for best results</li>
              <li>Keep your signature within the designated area</li>
              <li>Provide signatures from different angles and speeds</li>
            </ul>
          </div>
        </Alert>
      </CardContent>
    </Card>
  );
}
