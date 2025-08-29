import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, Camera, Play, StopCircle, CheckCircle, XCircle, Users, User, Clock, Calendar, BookOpen, ArrowLeft, RefreshCw, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import Layout from "@/components/Layout";
import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { supabase } from '@/lib/supabase';
import { aiService } from '@/lib/aiService';
import { toast } from 'sonner';

type Session = {
  id: number;
  title: string;
  description: string;
  program: string;
  year: string;
  section: string;
  date: string;
  time_in: string;
  time_out: string;
};

const TakeAttendanceSession = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [cameraActive, setCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [attendanceStatus, setAttendanceStatus] = useState<'success' | 'error' | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRequestingCamera, setIsRequestingCamera] = useState(false);
  const [cameraAvailable, setCameraAvailable] = useState<boolean | null>(null);
  const [stats, setStats] = useState({
    totalScanned: 0,
    matched: 0,
    noMatch: 0,
    potentialForgery: 0,
  });
  const [verificationResult, setVerificationResult] = useState<{
    match: boolean;
    student?: {
      id: number;
      student_id: string;
      firstname: string;
      surname: string;
    };
    score: number;
    message: string;
  } | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const permissionGranted = useRef(false);

  useEffect(() => {
    // Fetch session details when component mounts
    const fetchSession = async () => {
      try {
        setLoading(true);
        const { data, error } = await supabase
          .from('sessions')
          .select('*')
          .eq('id', parseInt(sessionId))
          .single();

        if (error) throw error;
        setSession(data);
      } catch (err) {
        console.error('Error fetching session:', err);
        setError('Failed to load session details');
      } finally {
        setLoading(false);
      }
    };

    if (sessionId) {
      fetchSession();
    } else {
      setLoading(false);
    }

    // Clean up camera stream when component unmounts
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
        });
        streamRef.current = null;
      }
      permissionGranted.current = false;
      setCameraActive(false);
    };
  }, [sessionId]);

  // Check for basic MediaDevices API support on component mount
  useEffect(() => {
    const checkBasicSupport = () => {
      // Define type for extended Navigator
      interface NavigatorExtended extends Navigator {
        webkitGetUserMedia?: typeof navigator.mediaDevices.getUserMedia;
        mozGetUserMedia?: typeof navigator.mediaDevices.getUserMedia;
        msGetUserMedia?: typeof navigator.mediaDevices.getUserMedia;
      }
      
      const nav = navigator as NavigatorExtended;
      const hasMediaDevices = !!(nav.mediaDevices && nav.mediaDevices.getUserMedia);
      const hasGetUserMedia = !!(nav.mediaDevices?.getUserMedia || 
                               nav.webkitGetUserMedia || 
                               nav.mozGetUserMedia ||
                               nav.msGetUserMedia);
      
      // Check if we're on a mobile device
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const isLocalNetwork = ['localhost', '127.0.0.1', '192.168.254.100'].includes(window.location.hostname);
      const isSecureContext = window.isSecureContext;
      
      console.log('Basic media devices check:', {
        hasMediaDevices,
        hasGetUserMedia,
        isSecureContext,
        isMobile,
        isLocalNetwork,
        location: window.location.href
      });
      
      // For mobile devices, we need HTTPS unless on localhost
      if (isMobile && !isSecureContext && !isLocalNetwork) {
        setCameraAvailable(false);
        return;
      }
      
      // Just check for basic API support, don't enumerate devices yet
      setCameraAvailable(hasMediaDevices || hasGetUserMedia);
    };
    
    checkBasicSupport();
  }, []);

  const handleStartCamera = async () => {
    console.log('Starting camera...');
    
    // Reset any previous error
    setError(null);
    
    if (cameraActive || !videoRef.current) {
      console.log('Camera already active or video ref not available');
      return;
    }
    
    setIsRequestingCamera(true);
    
    try {
      // First, check if we have permission to access media devices
      let devices: MediaDeviceInfo[] = [];
      try {
        devices = await navigator.mediaDevices.enumerateDevices();
        console.log('Available devices:', devices);
        
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        console.log('Video devices found:', videoDevices);
        
        if (videoDevices.length === 0) {
          throw new Error('No video input devices found');
        }
      } catch (err) {
        console.warn('Could not enumerate devices:', err);
        // Continue anyway, as some browsers might not support enumerateDevices
        // or might require getUserMedia first to get device labels
      }
      // Log environment info for debugging
      console.log('Environment info:', {
        userAgent: navigator.userAgent,
        isSecureContext: window.isSecureContext,
        location: window.location.href,
        mediaDevices: !!navigator.mediaDevices,
        getUserMedia: !!(navigator.mediaDevices?.getUserMedia)
      });
      
      // Check if we're on a mobile device and local network
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const isLocalNetwork = ['localhost', '127.0.0.1', '192.168.254.100'].includes(window.location.hostname);
      const isSecureContext = window.isSecureContext;
      
      // Check HTTPS requirement for mobile devices
      if (isMobile && !isSecureContext && !isLocalNetwork) {
        throw new Error('Camera access requires HTTPS on mobile devices. Please use HTTPS or access from localhost.');
      }
      
      if (isMobile && isLocalNetwork) {
        console.log('Mobile device on local network detected, using legacy API if needed');
      }
      console.log('Requesting camera access...');
      
      // Define the type for getUserMedia function
      type GetUserMediaFunction = (constraints: MediaStreamConstraints) => Promise<MediaStream>;
      let getUserMedia: GetUserMediaFunction;

      // Define types for legacy browser support
      interface NavigatorWithLegacyGetUserMedia extends Navigator {
        webkitGetUserMedia?: (
          constraints: MediaStreamConstraints,
          success: (stream: MediaStream) => void,
          error: (error: Error) => void
        ) => void;
        mozGetUserMedia?: (
          constraints: MediaStreamConstraints,
          success: (stream: MediaStream) => void,
          error: (error: Error) => void
        ) => void;
      }

      const navWithLegacy = navigator as NavigatorWithLegacyGetUserMedia;
      let stream: MediaStream | null = null;

      // Function to try getting media with specific constraints
      const tryGetMedia = async (constraints: MediaStreamConstraints): Promise<MediaStream> => {
        console.log('Trying constraints:', JSON.stringify(constraints));
        
        try {
          if (navWithLegacy.mediaDevices?.getUserMedia) {
            // Standard way (modern browsers)
            return await navWithLegacy.mediaDevices.getUserMedia(constraints);
          } else if (navWithLegacy.webkitGetUserMedia) {
            // Old Chrome/WebKit way
            return await new Promise<MediaStream>((resolve, reject) => 
              navWithLegacy.webkitGetUserMedia!(
                constraints,
                (s) => resolve(s),
                (e) => reject(e)
              )
            );
          } else if (navWithLegacy.mozGetUserMedia) {
            // Old Firefox way
            return await new Promise<MediaStream>((resolve, reject) => 
              navWithLegacy.mozGetUserMedia!(
                constraints,
                (s) => resolve(s),
                (e) => reject(e)
              )
            );
          }
          throw new Error('No supported camera API found');
        } catch (err) {
          console.error('Error in tryGetMedia:', err);
          throw err;
        }
      };

      // Try different constraint sets in sequence, prioritizing rear camera
      const constraintSets: MediaStreamConstraints[] = [
        // First try with rear camera (environment) and high resolution
        { 
          video: { 
            facingMode: { exact: 'environment' },
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          },
          audio: false 
        },
        // Fallback to any rear camera with default resolution
        { 
          video: { 
            facingMode: { exact: 'environment' }
          },
          audio: false 
        },
        // If no rear camera is found, try any camera
        { 
          video: true,
          audio: false 
        }
      ];
      
      console.log('Trying constraint sets:', constraintSets);

      // Try each constraint set until one works
      for (const constraints of constraintSets) {
        try {
          stream = await tryGetMedia(constraints);
          if (stream) {
            console.log('Successfully obtained stream with constraints:', constraints);
            break;
          }
        } catch (err) {
          console.warn(`Failed with constraints ${JSON.stringify(constraints)}:`, err);
        }
      }
      
      if (!stream) {
        // If we have video devices but couldn't access any, show a more specific error
        const hasVideoDevices = devices.some(d => d.kind === 'videoinput');
        if (hasVideoDevices) {
          throw new Error('Could not access camera. Please check browser permissions and ensure no other app is using the camera.');
        } else {
          throw new Error('No camera found. Please ensure a camera is connected and not in use by another application.');
        }
      }
      console.log('Successfully got media stream');
      
      // Set the video element's source
      if (videoRef.current) {
        // Stop any existing stream first
        if (streamRef.current) {
          console.log('Stopping existing stream');
          streamRef.current.getTracks().forEach(track => track.stop());
        }
        
        // Set the new stream
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        
        // Wait for the video to be ready to play
        return new Promise<void>((resolve) => {
          if (!videoRef.current) {
            setError('Video element not available');
            return resolve();
          }
          
          const onCanPlay = () => {
            console.log('Video can play, starting...');
            videoRef.current?.play()
              .then(() => {
                console.log('Video playback started');
                setCameraActive(true);
                setCapturedImage(null);
                setAttendanceStatus(null);
                resolve();
              })
              .catch(playErr => {
                console.error('Error playing video:', playErr);
                setError('Could not start the camera. Please try again.');
                handleStopCamera();
                resolve();
              });
          };
          
          // Set up event listeners
          videoRef.current.oncanplay = onCanPlay;
          videoRef.current.onerror = (err) => {
            console.error('Video element error:', err);
            setError('Error initializing video stream');
            resolve();
          };
          
          // If the video is already ready, call onCanPlay directly
          if (videoRef.current.readyState >= 2) {
            onCanPlay();
          }
        });
      }
      
    } catch (err) {
      console.error('Camera error:', err);
      
      if (err instanceof Error) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setError('Camera access was denied. Please check your browser settings and allow camera access.');
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          setError('No camera found on this device.');
        } else if (err.name === 'NotReadableError') {
          setError('Camera is already in use by another application.');
        } else if (err.name === 'TypeError' && err.message.includes('navigator.mediaDevices')) {
          setError('Your browser does not support camera access or is not in a secure context (try using HTTPS or localhost).');
        } else if (err.message.includes('HTTPS')) {
          setError(err.message);
        } else {
          setError(`Camera error: ${err.message}`);
        }
      } else {
        setError('Failed to access camera. Please check your browser settings and try again.');
      }
    } finally {
      setIsRequestingCamera(false);
    }
  };

  const handleStopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop();
      });
      streamRef.current = null;
    }
    setCameraActive(false);
    permissionGranted.current = false;
  };

  const handleCaptureSignature = async () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const imageUrl = canvas.toDataURL('image/png');
        setCapturedImage(imageUrl);
        
        // Stop the camera after capturing
        handleStopCamera();
        
        // Verify signature with AI
        await verifySignatureWithAI(imageUrl);
      }
    }
  };

  const verifySignatureWithAI = async (imageDataUrl: string) => {
    setIsVerifying(true);
    setVerificationResult(null);
    setAttendanceStatus(null);
    
    try {
      console.log('Starting AI signature verification...');
      
      // Call AI service for verification
      const result = await aiService.verifySignatureFromDataURL(
        imageDataUrl,
        sessionId ? parseInt(sessionId) : undefined
      );
      
      console.log('AI verification result:', result);
      
      // Update stats
      setStats(prev => ({
        ...prev,
        totalScanned: prev.totalScanned + 1,
        matched: prev.matched + (result.match ? 1 : 0),
        noMatch: prev.noMatch + (!result.match ? 1 : 0),
        potentialForgery: prev.potentialForgery + (result.decision === 'error' ? 1 : 0),
      }));
      
      if (result.success) {
        setVerificationResult({
          match: result.match,
          student: result.predicted_student,
          score: result.score,
          message: result.message,
        });
        
        if (result.match && result.predicted_student) {
          setAttendanceStatus('success');
          toast.success(`Signature matched: ${result.predicted_student.firstname} ${result.predicted_student.surname} (${result.predicted_student.student_id})`);
        } else {
          setAttendanceStatus('error');
          toast.warning('Signature not recognized. Please try again or mark attendance manually.');
        }
      } else {
        setVerificationResult({
          match: false,
          score: 0,
          message: result.error || 'Verification failed',
        });
        setAttendanceStatus('error');
        toast.error(result.error || 'Signature verification failed');
      }
      
    } catch (error) {
      console.error('Error during signature verification:', error);
      setVerificationResult({
        match: false,
        score: 0,
        message: 'Verification error occurred',
      });
      setAttendanceStatus('error');
      toast.error('Failed to verify signature. Please try again.');
      
      // Update stats for error
      setStats(prev => ({
        ...prev,
        totalScanned: prev.totalScanned + 1,
        potentialForgery: prev.potentialForgery + 1,
      }));
    } finally {
      setIsVerifying(false);
    }
  };

  const handleRetakeSignature = () => {
    setCapturedImage(null);
    setVerificationResult(null);
    setAttendanceStatus(null);
    // Don't restart camera automatically, let user click to start
  };

  if (loading) {
    return (
      <Layout>
        <div className="p-8">
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary mx-auto mb-4"></div>
            <h2 className="text-xl font-medium text-education-navy">
              Loading session details...
            </h2>
          </div>
        </div>
      </Layout>
    );
  }

  if (error || !session) {
    return (
      <Layout>
        <div className="p-8">
          <div className="text-center py-12">
            <h2 className="text-2xl font-bold text-education-navy mb-4">
              {error || 'Session not found'}
            </h2>
            <Button 
              onClick={() => navigate('/take-attendance')}
              className="mt-4"
            >
              Back to Sessions
            </Button>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="px-6 py-4 space-y-6">
        {/* Session Header - Left Aligned */}
        <div className="text-left space-y-1">
          <h1 className="text-3xl font-bold text-education-navy">{session.title}</h1>
          <p className="text-muted-foreground text-sm">
            {session.program} • {session.year} • {session.section} • {new Date(session.date).toLocaleDateString()} • {session.time_in} - {session.time_out}
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Section: Signature Capture */}
          <Card className="shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Camera className="w-5 h-5 text-education-navy" />
                Signature Capture
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col space-y-4 p-4">
              {/* Camera Preview Box - Always Visible */}
              <div className="w-full h-48 bg-muted/30 rounded-lg flex items-center justify-center border border-muted-foreground/20">
                <div className="w-full h-full relative">
                  {/* Video element - always present but only visible when active */}
                  <video 
                    ref={videoRef}
                    autoPlay 
                    playsInline 
                    muted 
                    className={`w-full h-full object-cover rounded-lg transition-opacity duration-300 ${
                      cameraActive ? 'opacity-100' : 'opacity-0 absolute'
                    }`}
                  />
                  
                  {/* Error state */}
                  {error && (
                    <div className="absolute inset-0 bg-red-50 rounded-lg flex items-center justify-center p-4">
                      <div className="text-center">
                        <XCircle className="w-12 h-12 text-red-400 mx-auto mb-2" />
                        <p className="text-red-700 font-medium">Camera Error</p>
                        <p className="text-sm text-red-600 max-w-xs">{error}</p>
                      </div>
                    </div>
                  )}
                  
                  {/* Captured image */}
                  {capturedImage && !cameraActive && !error && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <img 
                        src={capturedImage} 
                        alt="Captured Signature" 
                        className="w-full h-full object-cover rounded-lg" 
                      />
                    </div>
                  )}
                  
                  {/* Initial/Inactive state */}
                  {!cameraActive && !capturedImage && !error && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center p-4">
                      <div className="w-16 h-16 bg-muted/50 rounded-full flex items-center justify-center mb-4">
                        <Camera className="w-8 h-8 text-muted-foreground" />
                      </div>
                      <p className="text-muted-foreground text-center">
                        {isRequestingCamera ? 'Initializing camera...' : 'Camera not active'}
                      </p>
                    </div>
                  )}
                </div>
              </div>

                  {/* Camera Status Message */}
                  {cameraAvailable === false && (
                    <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-sm">
                      <p className="font-medium">Camera access restricted</p>
                      <p className="text-xs">
                        {!window.isSecureContext && !['localhost', '127.0.0.1', '192.168.254.100'].includes(window.location.hostname)
                          ? 'Camera access requires HTTPS on mobile devices. Please use HTTPS or access from localhost.'
                          : 'No camera was detected on this device or camera access is not supported.'
                        }
                      </p>
                    </div>
                  )}

                  {/* HTTPS Requirement Message for Mobile */}
                  {/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) && 
                   !window.isSecureContext && 
                   !['localhost', '127.0.0.1', '192.168.254.100'].includes(window.location.hostname) && (
                    <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-800">
                      <div className="flex items-start">
                        <div className="flex-shrink-0">
                          <XCircle className="w-5 h-5 mt-0.5" />
                        </div>
                        <div className="ml-3">
                          <h3 className="text-sm font-medium">HTTPS Required for Mobile Camera Access</h3>
                          <div className="mt-2 text-sm">
                            <p>Mobile browsers require HTTPS to access the camera. To use camera features:</p>
                            <ul className="list-disc list-inside mt-1 space-y-1">
                              <li>Use HTTPS instead of HTTP</li>
                              <li>Access from localhost (localhost:3000)</li>
                              <li>Use a secure development server</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Verification Results */}
                  {capturedImage && verificationResult && !isVerifying && (
                    <div className={cn(
                      "mb-4 p-4 rounded-lg border",
                      verificationResult.match 
                        ? "bg-green-50 border-green-200 text-green-800"
                        : "bg-red-50 border-red-200 text-red-800"
                    )}>
                      <div className="flex items-center mb-2">
                        {verificationResult.match ? (
                          <CheckCircle className="w-5 h-5 mr-2" />
                        ) : (
                          <XCircle className="w-5 h-5 mr-2" />
                        )}
                        <span className="font-medium">
                          {verificationResult.match ? 'Match Found!' : 'No Match'}
                        </span>
                      </div>
                      
                      {verificationResult.student && (
                        <div className="text-sm mb-2">
                          <strong>{verificationResult.student.firstname} {verificationResult.student.surname}</strong>
                          <br />
                          Student ID: {verificationResult.student.student_id}
                        </div>
                      )}
                      
                      <div className="text-sm">
                        Confidence: {Math.round(verificationResult.score * 100)}%
                      </div>
                      
                      {verificationResult.message && (
                        <div className="text-xs mt-2 opacity-80">
                          {verificationResult.message}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Verification Loading */}
                  {isVerifying && (
                    <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg text-blue-800">
                      <div className="flex items-center">
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        <span className="font-medium">Verifying signature...</span>
                      </div>
                      <div className="text-sm mt-1 opacity-80">
                        Please wait while we process your signature
                      </div>
                    </div>
                  )}

                  {/* Button Group */}
                  <div className="flex flex-col space-y-2">
                    {!cameraActive && !capturedImage ? (
                      <Button 
                        onClick={handleStartCamera}
                        disabled={isRequestingCamera || cameraAvailable === false}
                        className="w-full bg-teal-300 text-white hover:bg-teal-200 hover:text-teal-900 py-2 h-auto text-base transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                        size="lg"
                      >
                        {isRequestingCamera ? (
                          <>
                            <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                            <span>Requesting Access...</span>
                          </>
                        ) : (
                          <>
                            <Play className="w-5 h-5 mr-2 text-white" />
                            <span className="text-white">
                              {cameraAvailable === false ? 'Camera Not Available' : 'Start Camera'}
                            </span>
                          </>
                        )}
                      </Button>
                    ) : cameraActive ? (
                      <div className="flex gap-2 w-full">
                        <Button 
                          onClick={handleCaptureSignature}
                          disabled={isVerifying}
                          className="flex-1 bg-teal-300 text-white hover:bg-teal-200 hover:text-teal-900 py-2 h-auto text-base transition-all duration-200 disabled:opacity-50"
                          size="lg"
                        >
                          <Camera className="w-5 h-5 mr-2 text-white" />
                          <span className="text-white">Capture Signature</span>
                        </Button>
                        <Button 
                          onClick={handleStopCamera}
                          variant="outline"
                          className="flex-shrink-0 px-3 sm:px-4 py-2 h-auto text-base hover:bg-transparent hover:text-inherit active:bg-transparent focus:bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0"
                          size="lg"
                        >
                          <div className="relative w-5 h-5 mr-2">
                            <Square className="w-4 h-4 text-red-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                            <div className="w-3 h-3 bg-red-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                          </div>
                          <span className="hidden sm:inline">Stop</span>
                        </Button>
                      </div>
                    ) : capturedImage ? (
                      <div className="flex gap-2 w-full">
                        <Button 
                          onClick={handleRetakeSignature}
                          variant="outline"
                          className="flex-1 py-2 h-auto text-base"
                          size="lg"
                          disabled={isVerifying}
                        >
                          <RefreshCw className="w-5 h-5 mr-2" />
                          Retake
                        </Button>
                        <Button 
                          onClick={handleStartCamera}
                          className="flex-1 bg-teal-300 text-white hover:bg-teal-200 hover:text-teal-900 py-2 h-auto text-base transition-all duration-200"
                          size="lg"
                          disabled={isVerifying}
                        >
                          <Camera className="w-5 h-5 mr-2 text-white" />
                          <span className="text-white">New Capture</span>
                        </Button>
                      </div>
                    ) : null}
                  </div>
            </CardContent>
          </Card>

          {/* Right Section: Attendance Log */}
          <Card className="shadow-sm">
            <CardHeader className="p-4 pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Attendance Log</CardTitle>
                <Badge className="bg-green-100 text-green-800 hover:bg-green-100">
                  {stats.matched} captured
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="w-full h-48 bg-muted/10 flex flex-col items-center justify-center">
                <Users className="w-12 h-12 text-muted-300 mb-2" />
                <p className="text-muted-500">No signatures captured yet</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Statistics Section */}
        <Card className="shadow-sm mt-6">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Session Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {/* Total Scanned */}
              <div className="bg-muted/30 p-4 rounded-lg">
                <div className="text-2xl font-bold text-center">{stats.totalScanned}</div>
                <p className="text-center text-muted-foreground text-sm mt-1">Total Scanned</p>
              </div>

              {/* Matched */}
              <div className="bg-muted/30 p-4 rounded-lg">
                <div className="text-2xl font-bold text-center text-green-600">
                  {stats.matched}
                </div>
                <p className="text-center text-muted-foreground text-sm mt-1">Matched</p>
              </div>

              {/* No Match */}
              <div className="bg-muted/30 p-4 rounded-lg">
                <div className="text-2xl font-bold text-center text-red-600">
                  {stats.noMatch}
                </div>
                <p className="text-center text-muted-foreground text-sm mt-1">No Match</p>
              </div>

              {/* Potential Forgery */}
              <div className="bg-muted/30 p-4 rounded-lg">
                <div className="text-2xl font-bold text-center text-amber-500">
                  {stats.potentialForgery}
                </div>
                <p className="text-center text-muted-foreground text-sm mt-1">Potential Forgery</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
};

export default TakeAttendanceSession;