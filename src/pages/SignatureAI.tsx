import React, { useState, useRef } from 'react';
import Layout from '@/components/Layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { FileUpload } from '@/components/ui/file-upload';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/components/ui/use-toast';
import { 
  Upload, 
  Camera, 
  Brain, 
  Scan, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  Loader2
} from 'lucide-react';
import { aiService } from '@/lib/aiService';

interface TrainingFile {
  file: File;
  preview: string;
}

const SignatureAI = () => {
  const { toast } = useToast();
  
  // Training Section State
  const [trainingFiles, setTrainingFiles] = useState<TrainingFile[]>([]);
  const [studentId, setStudentId] = useState<string>('');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<any>(null);
  
  // Verification Section State
  const [verificationFile, setVerificationFile] = useState<File | null>(null);
  const [verificationPreview, setVerificationPreview] = useState<string>('');
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<any>(null);
  const [useCamera, setUseCamera] = useState(false);
  
  const verificationInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Training Functions
  const handleTrainingFilesChange = (files: File[]) => {
    const newFiles = files.map(file => ({
      file,
      preview: URL.createObjectURL(file)
    }));
    setTrainingFiles(prev => [...prev, ...newFiles]);
  };

  const removeTrainingFile = (index: number) => {
    setTrainingFiles(prev => {
      const newFiles = [...prev];
      URL.revokeObjectURL(newFiles[index].preview);
      newFiles.splice(index, 1);
      return newFiles;
    });
  };

  const handleTrainModel = async () => {
    if (!studentId.trim()) {
      toast({
        title: "Error",
        description: "Please enter a student ID",
        variant: "destructive",
      });
      return;
    }

    if (trainingFiles.length === 0) {
      toast({
        title: "Error", 
        description: "Please upload at least one signature sample",
        variant: "destructive",
      });
      return;
    }

    setIsTraining(true);
    setTrainingResult(null);

    try {
      const result = await aiService.trainStudent(parseInt(studentId));
      setTrainingResult(result);
      
      if (result.success) {
        toast({
          title: "Training Started",
          description: "AI model training has been initiated for this student",
        });
      } else {
        toast({
          title: "Training Failed",
          description: result.message || "Failed to start training",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Training error:', error);
      toast({
        title: "Error",
        description: "An unexpected error occurred during training",
        variant: "destructive",
      });
    } finally {
      setIsTraining(false);
    }
  };

  // Verification Functions
  const handleVerificationFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVerificationFile(file);
      setVerificationPreview(URL.createObjectURL(file));
      setVerificationResult(null);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setUseCamera(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      toast({
        title: "Camera Error",
        description: "Unable to access camera. Please check permissions.",
        variant: "destructive",
      });
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const context = canvas.getContext('2d');
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      if (context) {
        context.drawImage(video, 0, 0);
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], 'signature.png', { type: 'image/png' });
            setVerificationFile(file);
            setVerificationPreview(URL.createObjectURL(file));
            setUseCamera(false);
            setVerificationResult(null);
            
            // Stop camera stream
            const stream = video.srcObject as MediaStream;
            stream?.getTracks().forEach(track => track.stop());
          }
        });
      }
    }
  };

  const stopCamera = () => {
    if (videoRef.current) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream?.getTracks().forEach(track => track.stop());
      setUseCamera(false);
    }
  };

  const handleVerifySignature = async () => {
    if (!verificationFile) {
      toast({
        title: "Error",
        description: "Please upload or capture a signature image",
        variant: "destructive",
      });
      return;
    }

    setIsVerifying(true);
    setVerificationResult(null);

    try {
      const result = await aiService.verifySignature(verificationFile);
      setVerificationResult(result);
      
      if (result.success) {
        toast({
          title: "Verification Complete",
          description: `Result: ${result.match ? 'Match found' : 'No match'}`,
        });
      } else {
        toast({
          title: "Verification Failed",
          description: result.message || "Failed to verify signature",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Verification error:', error);
      toast({
        title: "Error",
        description: "An unexpected error occurred during verification",
        variant: "destructive",
      });
    } finally {
      setIsVerifying(false);
    }
  };

  return (
    <Layout>
      <div className="flex-1 flex flex-col space-y-6 px-6 py-4">
        {/* Page Header */}
        <div className="space-y-0.5">
          <h1 className="text-lg font-bold text-education-navy">SIGNATURE AI</h1>
          <p className="text-sm text-muted-foreground">
            Train AI models and verify signatures using machine learning
          </p>
        </div>

        {/* Main Content - Two Cards Side by Side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Left Card: Model Training Section */}
          <Card className="h-fit">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5" />
                Model Training
              </CardTitle>
              <CardDescription>
                Upload signature samples to train AI model for a specific student
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Student ID Input */}
              <div className="space-y-2">
                <Label htmlFor="studentId">Student ID</Label>
                <Input
                  id="studentId"
                  type="number"
                  placeholder="Enter student ID"
                  value={studentId}
                  onChange={(e) => setStudentId(e.target.value)}
                />
              </div>

              <Separator />

              {/* Large Square Preview Box for Training Images */}
              <div className="space-y-2">
                <Label>Training Images Preview</Label>
                <div className="w-full h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-gray-50">
                  {trainingFiles.length > 0 ? (
                    <div className="grid grid-cols-2 gap-2 w-full h-full p-4 overflow-y-auto">
                      {trainingFiles.map((item, index) => (
                        <div key={index} className="relative group">
                          <img
                            src={item.preview}
                            alt={`Sample ${index + 1}`}
                            className="w-full h-20 object-cover rounded border"
                          />
                          <Button
                            size="sm"
                            variant="destructive"
                            className="absolute top-1 right-1 w-6 h-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={() => removeTrainingFile(index)}
                          >
                            ×
                          </Button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center text-gray-500">
                      <Upload className="w-8 h-8 mx-auto mb-2" />
                      <p>No training images uploaded</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Upload Button */}
              <Button
                onClick={() => {
                  const input = document.createElement('input');
                  input.type = 'file';
                  input.accept = 'image/*';
                  input.multiple = true;
                  input.onchange = (e) => {
                    const files = Array.from((e.target as HTMLInputElement).files || []);
                    handleTrainingFilesChange(files);
                  };
                  input.click();
                }}
                variant="outline"
                className="w-full"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload Training Images
              </Button>

              {/* Train Button */}
              <Button
                onClick={handleTrainModel}
                disabled={!studentId.trim() || trainingFiles.length === 0 || isTraining}
                className="w-full"
              >
                {isTraining ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Training Model...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4 mr-2" />
                    Train Model
                  </>
                )}
              </Button>

              {/* Training Result */}
              {trainingResult && (
                <Alert className={trainingResult.success ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}>
                  <div className="flex items-center gap-2">
                    {trainingResult.success ? (
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-red-600" />
                    )}
                    <AlertDescription>
                      <div className="space-y-2">
                        <p>
                          <strong>Status:</strong> {trainingResult.success ? 'Training Started' : 'Training Failed'}
                        </p>
                        {trainingResult.profile && (
                          <>
                            <p>
                              <strong>Model Status:</strong>{' '}
                              <Badge variant={trainingResult.profile.status === 'ready' ? 'default' : 'secondary'}>
                                {trainingResult.profile.status}
                              </Badge>
                            </p>
                            <p>
                              <strong>Samples:</strong> {trainingResult.profile.num_samples}
                            </p>
                            {trainingResult.profile.last_trained_at && (
                              <p>
                                <strong>Last Trained:</strong>{' '}
                                {new Date(trainingResult.profile.last_trained_at).toLocaleString()}
                              </p>
                            )}
                          </>
                        )}
                        <p className="text-sm text-muted-foreground">
                          {trainingResult.message}
                        </p>
                      </div>
                    </AlertDescription>
                  </div>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Right Card: Signature Verification Section */}
          <Card className="h-fit">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Scan className="w-5 h-5" />
                Signature Verification
              </CardTitle>
              <CardDescription>
                Upload or capture a signature to verify against trained models
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Camera/Upload Toggle */}
              <div className="flex gap-2">
                <Button
                  variant={useCamera ? "default" : "outline"}
                  size="sm"
                  onClick={startCamera}
                  className="flex items-center gap-2"
                >
                  <Camera className="w-4 h-4" />
                  Camera
                </Button>
                <Button
                  variant={!useCamera ? "default" : "outline"}
                  size="sm"
                  onClick={() => {
                    setUseCamera(false);
                    stopCamera();
                    verificationInputRef.current?.click();
                  }}
                  className="flex items-center gap-2"
                >
                  <Upload className="w-4 h-4" />
                  Upload
                </Button>
              </div>

              {/* Large Square Preview Box */}
              <div className="space-y-2">
                <Label>Signature Preview</Label>
                <div className="w-full h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-gray-50">
                  {useCamera ? (
                    <div className="w-full h-full">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        className="w-full h-full object-cover rounded-lg"
                      />
                    </div>
                  ) : verificationPreview ? (
                    <img
                      src={verificationPreview}
                      alt="Signature preview"
                      className="w-full h-full object-contain rounded-lg"
                    />
                  ) : (
                    <div className="text-center text-gray-500">
                      <Upload className="w-8 h-8 mx-auto mb-2" />
                      <p>No signature selected</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Camera Controls */}
              {useCamera && (
                <div className="flex gap-2">
                  <Button onClick={capturePhoto} size="sm" className="flex-1">
                    Capture
                  </Button>
                  <Button onClick={stopCamera} variant="outline" size="sm" className="flex-1">
                    Cancel
                  </Button>
                </div>
              )}

              {/* Hidden File Input */}
              <Input
                ref={verificationInputRef}
                type="file"
                accept="image/*"
                onChange={handleVerificationFileChange}
                className="hidden"
              />

              {/* Verify Button */}
              <Button
                onClick={handleVerifySignature}
                disabled={!verificationFile || isVerifying}
                className="w-full"
              >
                {isVerifying ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Verifying...
                  </>
                ) : (
                  <>
                    <Scan className="w-4 h-4 mr-2" />
                    Verify Signature
                  </>
                )}
              </Button>

              {/* Verification Result */}
              {verificationResult && (
                <Alert className={verificationResult.match ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}>
                  <div className="flex items-center gap-2">
                    {verificationResult.match ? (
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-600" />
                    )}
                    <AlertDescription>
                      <div className="space-y-2">
                        <p>
                          <strong>Result:</strong> {verificationResult.match ? 'Match Found' : 'No Match'}
                        </p>
                        <p>
                          <strong>Confidence:</strong> {(verificationResult.score * 100).toFixed(1)}%
                        </p>
                        {verificationResult.predicted_student && (
                          <p>
                            <strong>Predicted Student:</strong> {verificationResult.predicted_student.firstname} {verificationResult.predicted_student.surname}
                          </p>
                        )}
                        <p className="text-sm text-muted-foreground">
                          {verificationResult.message}
                        </p>
                      </div>
                    </AlertDescription>
                  </div>
                </Alert>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
};

export default SignatureAI;