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
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { useToast } from '@/components/ui/use-toast';
import { 
  Upload, 
  Camera, 
  Brain, 
  Scan, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  Loader2,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  User,
  MoreVertical,
  Trash2
} from 'lucide-react';
import { aiService } from '@/lib/aiService';

interface TrainingFile {
  file: File;
  preview: string;
}

const SignatureAI = () => {
  const { toast } = useToast();
  
  // Training Section State
  const [genuineFiles, setGenuineFiles] = useState<TrainingFile[]>([]);
  const [forgedFiles, setForgedFiles] = useState<TrainingFile[]>([]);
  const [currentTrainingSet, setCurrentTrainingSet] = useState<'genuine' | 'forged'>('genuine');
  const [studentId, setStudentId] = useState<string>('');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<any>(null);
  
  // Verification Section State
  const [verificationFile, setVerificationFile] = useState<File | null>(null);
  const [verificationPreview, setVerificationPreview] = useState<string>('');
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<any>(null);
  const [useCamera, setUseCamera] = useState(false);
  
  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalImageIndex, setModalImageIndex] = useState(0);
  const [modalImages, setModalImages] = useState<string[]>([]);
  
  // Dropdown State
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  // Student Selection State
  interface SelectedStudent {
    id: string;
    name: string;
    program: string;
    year: string;
    section: string;
  }
  const [selectedStudent, setSelectedStudent] = useState<SelectedStudent | null>(null);
  const [isStudentDialogOpen, setIsStudentDialogOpen] = useState(false);
  const [studentSearch, setStudentSearch] = useState('');
  const [isStudentCollapsed, setIsStudentCollapsed] = useState(false);
  // Mock students list for UI demonstration; replace with API in integration
  const mockStudents: SelectedStudent[] = [
    { id: '2023001', name: 'Juan Dela Cruz', program: 'BSIT', year: '3', section: 'A' },
    { id: '2023002', name: 'Maria Santos', program: 'BSCS', year: '2', section: 'B' },
    { id: '2023003', name: 'John Reyes', program: 'BSIS', year: '1', section: 'C' },
  ];
  const filteredStudents = mockStudents.filter(s => 
    s.id.includes(studentSearch.trim()) || s.name.toLowerCase().includes(studentSearch.trim().toLowerCase())
  );
  
  const verificationInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Training Functions
  const handleTrainingFilesChange = (files: File[], setType: 'genuine' | 'forged') => {
    const newFiles = files.map(file => ({
      file,
      preview: URL.createObjectURL(file)
    }));
    if (setType === 'genuine') {
      setGenuineFiles(prev => [...prev, ...newFiles]);
    } else {
      setForgedFiles(prev => [...prev, ...newFiles]);
    }
  };

  const removeTrainingFile = (index: number, setType: 'genuine' | 'forged') => {
    if (setType === 'genuine') {
      setGenuineFiles(prev => {
        const newFiles = [...prev];
        URL.revokeObjectURL(newFiles[index].preview);
        newFiles.splice(index, 1);
        return newFiles;
      });
    } else {
      setForgedFiles(prev => {
        const newFiles = [...prev];
        URL.revokeObjectURL(newFiles[index].preview);
        newFiles.splice(index, 1);
        return newFiles;
      });
    }
  };

  const handleTrainModel = async () => {
    if (!studentId.trim()) {
      toast({
        title: "Error",
        description: "Please select a student",
        variant: "destructive",
      });
      return;
    }

    if ((genuineFiles.length + forgedFiles.length) === 0) {
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

  // Modal Functions
  type ModalContext = { kind: 'training', setType: 'genuine' | 'forged' } | { kind: 'verification' } | null;
  const [modalContext, setModalContext] = useState<ModalContext>(null);

  const openImageModal = (images: string[], startIndex: number = 0, context: ModalContext = null) => {
    setModalImages(images);
    setModalImageIndex(startIndex);
    setIsModalOpen(true);
    setModalContext(context);
  };

  const closeImageModal = () => {
    setIsModalOpen(false);
    setModalImages([]);
    setModalImageIndex(0);
  };

  const goToPreviousImage = () => {
    setModalImageIndex(prev => prev > 0 ? prev - 1 : modalImages.length - 1);
  };

  const goToNextImage = () => {
    setModalImageIndex(prev => prev < modalImages.length - 1 ? prev + 1 : 0);
  };

  const removeAllTrainingFiles = () => {
    genuineFiles.forEach(file => URL.revokeObjectURL(file.preview));
    forgedFiles.forEach(file => URL.revokeObjectURL(file.preview));
    setGenuineFiles([]);
    setForgedFiles([]);
    setIsDropdownOpen(false);
    toast({
      title: "Samples Removed",
      description: "All training samples have been removed",
    });
  };

  const deleteModalCurrentImage = () => {
    if (!modalContext || modalContext.kind !== 'training') return;
    const targetPreview = modalImages[modalImageIndex];
    if (modalContext.setType === 'genuine') {
      const idx = genuineFiles.findIndex(f => f.preview === targetPreview);
      if (idx !== -1) removeTrainingFile(idx, 'genuine');
      const updated = genuineFiles.filter(f => f.preview !== targetPreview).map(f => f.preview);
      setModalImages(updated);
    } else {
      const idx = forgedFiles.findIndex(f => f.preview === targetPreview);
      if (idx !== -1) removeTrainingFile(idx, 'forged');
      const updated = forgedFiles.filter(f => f.preview !== targetPreview).map(f => f.preview);
      setModalImages(updated);
    }
    setModalImageIndex(prev => Math.max(0, prev - (modalImages.length === 1 ? 0 : 1)));
    if (modalImages.length <= 1) closeImageModal();
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

        {/* Student Selection Card */}
        <Card>
          <CardHeader className="py-3">
            {/* Collapsed header */}
            {isStudentCollapsed ? (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-left">
                  <span className="text-sm font-medium">
                    {selectedStudent?.name ?? 'No student selected'}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    className={`h-6 w-6 p-0 opacity-60 text-muted-foreground hover:opacity-100 hover:text-foreground hover:bg-transparent transition-transform ${isStudentCollapsed ? 'rotate-180' : ''}`}
                    onClick={() => setIsStudentCollapsed(false)}
                    aria-label="Expand"
                  >
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0 opacity-60 text-muted-foreground hover:opacity-100 hover:text-foreground hover:bg-transparent transition-colors"
                      >
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => setIsStudentDialogOpen(true)}>
                        {selectedStudent ? 'Change' : 'Select'}
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
            ) : (
              // Expanded header
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <CardTitle className="flex items-center gap-2">
                    Select Student
                  </CardTitle>
                  <CardDescription>
                    Choose a student to train the AI model for signature verification
                  </CardDescription>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    className={`h-6 w-6 p-0 opacity-60 text-muted-foreground hover:opacity-100 hover:text-foreground hover:bg-transparent transition-transform ${isStudentCollapsed ? 'rotate-180' : ''}`}
                    onClick={() => setIsStudentCollapsed(true)}
                    aria-label="Collapse"
                  >
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0 opacity-60 text-muted-foreground hover:opacity-100 hover:text-foreground hover:bg-transparent transition-colors"
                      >
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => setIsStudentDialogOpen(true)}>
                        {selectedStudent ? 'Change' : 'Select'}
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
            )}
          </CardHeader>
          {/* Collapsible content */}
          <div className={`overflow-hidden transition-all ${isStudentCollapsed ? 'max-h-0 py-0' : 'max-h-[500px]'}`}>
            {!isStudentCollapsed && (
              <CardContent>
                {selectedStudent ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-2xl">
                    <div>
                      <Label className="text-muted-foreground">ID</Label>
                      <div className="font-medium">{selectedStudent?.id ?? '—'}</div>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Name</Label>
                      <div className="font-medium">{selectedStudent?.name ?? '—'}</div>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Program</Label>
                      <div className="font-medium">{selectedStudent?.program ?? '—'}</div>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Year</Label>
                      <div className="font-medium">{selectedStudent?.year ?? '—'}</div>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Section</Label>
                      <div className="font-medium">{selectedStudent?.section ?? '—'}</div>
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-muted-foreground">No student selected.</div>
                )}
              </CardContent>
            )}
          </div>
        </Card>

        {/* Main Content - Two Cards Side by Side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Left Card: Model Training Section */}
          <Card className="h-fit">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    Model Training
                  </CardTitle>
                  <CardDescription>
                    Upload signature samples to train AI model for a specific student
                  </CardDescription>
                </div>
                <DropdownMenu open={isDropdownOpen} onOpenChange={setIsDropdownOpen}>
                  <DropdownMenuTrigger asChild>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="h-6 w-6 p-0 opacity-40 hover:opacity-100 hover:bg-transparent transition-opacity"
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem 
                      onClick={removeAllTrainingFiles} 
                      className="text-red-600"
                      disabled={(genuineFiles.length + forgedFiles.length) === 0}
                    >
                      Remove All Samples
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Upload Buttons: Forged (left, normal) and Genuine (right, highlighted) */}
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-2 hover:bg-transparent hover:text-foreground"
                  onClick={() => {
                    setCurrentTrainingSet('forged');
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.accept = 'image/*';
                    input.multiple = true;
                    input.onchange = (e) => {
                      const files = Array.from((e.target as HTMLInputElement).files || []);
                      handleTrainingFilesChange(files, 'forged');
                    };
                    input.click();
                  }}
                >
                  <Upload className="w-4 h-4" />
                  Forged
                </Button>
                <Button
                  variant="default"
                  size="sm"
                  className="flex items-center gap-2"
                  onClick={() => {
                    setCurrentTrainingSet('genuine');
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.accept = 'image/*';
                    input.multiple = true;
                    input.onchange = (e) => {
                      const files = Array.from((e.target as HTMLInputElement).files || []);
                      handleTrainingFilesChange(files, 'genuine');
                    };
                    input.click();
                  }}
                >
                  <Upload className="w-4 h-4" />
                  Genuine
                </Button>
              </div>

              {/* Large Square Preview Box for Training Images (Genuine/Forged switch) */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Training Images Preview</Label>
                  <div className="text-xs text-muted-foreground">
                    {currentTrainingSet === 'genuine' ? (
                      <span>Genuine ({genuineFiles.length})</span>
                    ) : (
                      <span>Forged ({forgedFiles.length})</span>
                    )}
                  </div>
                </div>
                <div className="relative w-full h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-gray-50 group">
                  {/* Hover Previous/Next inside box */}
                  <button
                    className="hidden group-hover:flex absolute left-2 top-1/2 -translate-y-1/2 bg-white/80 hover:bg-white rounded-full p-2 shadow"
                    onClick={() => setCurrentTrainingSet(prev => prev === 'genuine' ? 'forged' : 'genuine')}
                    aria-label="Previous"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <button
                    className="hidden group-hover:flex absolute right-2 top-1/2 -translate-y-1/2 bg-white/80 hover:bg-white rounded-full p-2 shadow"
                    onClick={() => setCurrentTrainingSet(prev => prev === 'genuine' ? 'forged' : 'genuine')}
                    aria-label="Next"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>

                  {((currentTrainingSet === 'genuine' ? genuineFiles : forgedFiles).length > 0) ? (
                    <div className="grid grid-cols-3 gap-2 w-full h-full p-4 overflow-y-auto">
                      {(currentTrainingSet === 'genuine' ? genuineFiles : forgedFiles).map((item, index) => (
                        <div key={index} className="relative group/itm cursor-pointer" onClick={() => openImageModal((currentTrainingSet === 'genuine' ? genuineFiles : forgedFiles).map(f => f.preview), index, { kind: 'training', setType: currentTrainingSet })}>
                          <img
                            src={item.preview}
                            alt={`Sample ${index + 1}`}
                            className="w-full h-16 object-cover rounded border hover:opacity-80 transition-opacity"
                          />
                          <Button
                            size="sm"
                            variant="destructive"
                            className="absolute top-1 right-1 w-6 h-6 p-0 opacity-0 group-hover/itm:opacity-100 transition-opacity flex items-center justify-center"
                            onClick={(e) => {
                              e.stopPropagation();
                              removeTrainingFile(index, currentTrainingSet);
                            }}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center text-gray-500">
                      <Upload className="w-8 h-8 mx-auto mb-2" />
                      <p>No {currentTrainingSet === 'genuine' ? 'genuine' : 'forged'} images uploaded</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Train Button */}
              <Button
                onClick={handleTrainModel}
                disabled={!studentId.trim() || (genuineFiles.length + forgedFiles.length) === 0 || isTraining}
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
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <CardTitle className="flex items-center gap-2">
                    <Scan className="w-5 h-5" />
                    Signature Verification
                  </CardTitle>
                  <CardDescription>
                    Upload or capture a signature to verify against trained models
                  </CardDescription>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 opacity-40 hover:opacity-100 hover:bg-transparent transition-opacity"
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      onClick={() => {
                        setVerificationFile(null);
                        setVerificationPreview('');
                        stopCamera();
                      }}
                      disabled={!verificationFile && !useCamera && !verificationPreview}
                    >
                      Clear
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
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
                <div className="flex items-center justify-between">
                  <Label>Signature Preview</Label>
                  <div className="text-xs text-muted-foreground">{useCamera ? 'Camera' : 'Upload'}</div>
                </div>
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
                      className="w-full h-full object-contain rounded-lg cursor-pointer hover:opacity-80 transition-opacity"
                      onClick={() => openImageModal([verificationPreview], 0)}
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

        {/* Image Preview Modal */
        }
        <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
          <DialogContent className="max-w-4xl max-h-[90vh] p-0">
            <DialogHeader className="p-6 pb-0">
              <DialogTitle>Image Preview</DialogTitle>
            </DialogHeader>
            <div className="relative p-6">
              {modalImages.length > 0 && (
                <>
                  <img
                    src={modalImages[modalImageIndex]}
                    alt={`Preview ${modalImageIndex + 1}`}
                    className="w-full h-auto max-h-[60vh] object-contain mx-auto"
                  />
                  
                  {/* Prev/Next Arrows */}
                  <Button
                    variant="outline"
                    size="icon"
                    className="absolute left-4 top-1/2 transform -translate-y-1/2"
                    onClick={goToPreviousImage}
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    className="absolute right-4 top-1/2 transform -translate-y-1/2"
                    onClick={goToNextImage}
                  >
                    <ChevronRight className="w-4 h-4" />
                  </Button>

                  {/* Count + Trash inline with translucent background (old style) */}
                  <div className="absolute bottom-4 left-1/2 -translate-x-1/2">
                    <div className="bg-black/50 text-white px-3 py-1 rounded-full text-sm flex items-center gap-3">
                      <span>{modalImageIndex + 1} / {modalImages.length}</span>
                      {modalContext && modalContext.kind === 'training' && (
                        <button onClick={deleteModalCurrentImage} aria-label="Delete Image" className="text-white">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          </DialogContent>
        </Dialog>

        {/* Student Selection Dialog */}
        <Dialog open={isStudentDialogOpen} onOpenChange={setIsStudentDialogOpen}>
          <DialogContent className="max-w-xl">
            <DialogHeader>
              <DialogTitle>Select Student</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <Input
                placeholder="Search by ID or Name"
                value={studentSearch}
                onChange={(e) => setStudentSearch(e.target.value)}
              />
              <div className="max-h-64 overflow-auto border rounded-md">
                {filteredStudents.length === 0 ? (
                  <div className="p-4 text-sm text-muted-foreground">No results</div>
                ) : (
                  <ul className="divide-y">
                    {filteredStudents.map((s) => (
                      <li key={s.id ?? Math.random()}>
                        <button
                          className="w-full text-left p-3 hover:bg-muted/50"
                          onClick={() => {
                            setSelectedStudent(s);
                            if (s && s.id) setStudentId(s.id);
                            setIsStudentDialogOpen(false);
                          }}
                        >
                          <div className="font-medium">{s?.name ?? 'Unknown'}</div>
                          <div className="text-xs text-muted-foreground">ID: {s?.id ?? '—'} • {s?.program ?? '—'} • Year {s?.year ?? '—'} • Sec {s?.section ?? '—'}</div>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </Layout>
  );
};

export default SignatureAI;