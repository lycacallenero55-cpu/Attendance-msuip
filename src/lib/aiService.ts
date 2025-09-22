// AI Service for signature processing and training
// Focus: Owner identification only (no security detection)

const AI_SERVICE_URL = 'http://localhost:8000';

// Type definitions for API responses
interface StudentSignature {
  student_id: number;
  label: 'genuine';
  s3_url: string;
  s3_key?: string;
}

interface StudentWithImages {
  student_id: number;
  genuine_count?: number;
  signatures?: StudentSignature[];
}

interface StudentsWithImagesResponse {
  items: StudentWithImages[];
}

class AIService {
  private baseUrl = AI_SERVICE_URL;

  // Helper to get backend URL
  private getUrl(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  // Health check
  async healthCheck() {
    const res = await fetch(this.getUrl('/health'));
    if (!res.ok) throw new Error('AI service not available');
    return res.json();
  }

  // Preview generation for images
  async getPreviewURL(file: File): Promise<string> {
    if (file.type.startsWith('image/') && file.type !== 'image/tiff' && 
        !file.name.toLowerCase().endsWith('.tif') && !file.name.toLowerCase().endsWith('.tiff')) {
      return URL.createObjectURL(file);
    }
    
    // For non-image files, return a placeholder
    return '/placeholder-signature.png';
  }

  // S3-backed student signatures
  async uploadSignature(studentId: number, label: 'genuine', file: File) {
    const form = new FormData();
    form.append('student_id', String(studentId));
    form.append('label', label);
    form.append('file', file);
    const res = await fetch(this.getUrl('/api/uploads/signature'), { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    return data.record as { id:number; student_id:number; label:'genuine'; s3_url:string; s3_key:string };
  }

  async listSignatures(studentId: number) {
    const res = await fetch(this.getUrl(`/api/uploads/list?student_id=${studentId}`));
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'List failed');
    return data.signatures as Array<{ id:number; student_id:number; label:'genuine'; s3_url:string; s3_key:string }>;
  }

  async deleteSignature(signatureId: number, s3Key?: string) {
    const url = s3Key 
      ? this.getUrl(`/api/uploads/signature/${signatureId}?s3_key=${encodeURIComponent(s3Key)}`)
      : this.getUrl(`/api/uploads/signature/${signatureId}`);
    const res = await fetch(url, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Delete failed');
    return data;
  }

  // Training operations
  async listStudentsWithImages(): Promise<StudentWithImages[]> {
    const res = await fetch(this.getUrl('/api/uploads/students-with-images'));
    
    if (!res.ok) {
      const errorData = await res.json() as { detail?: string };
      throw new Error(errorData.detail || 'List failed');
    }
    
    const data: StudentsWithImagesResponse = await res.json();
    
    // Transform the response to match expected format
    const items = data.items || [];
    return items.map((item: StudentWithImages) => ({
      student_id: item.student_id,
      genuine_count: item.signatures ? item.signatures.filter((s: StudentSignature) => s.label === 'genuine').length : 0,
      signatures: item.signatures || []
    }));
  }

  // GPU-accelerated training with S3 storage
  async startGPUTraining(
    studentId: string,
    genuineFiles: File[],
    useGPU: boolean = true,
    saveToS3: boolean = true
  ): Promise<{
    job_id: string;
    message: string;
    student_id: string;
    training_mode: string;
  }> {
    try {
      const formData = new FormData();
      formData.append('student_id', studentId);
      formData.append('use_gpu', String(useGPU));
      formData.append('save_to_s3', String(saveToS3));
      
      for (const file of genuineFiles) {
        formData.append('genuine_files', file);
      }
      // Force global mode
      formData.append('training_mode', 'global');

      const response = await fetch(`${this.baseUrl}/api/training/start-gpu-training`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `GPU training failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('GPU training error:', error);
      throw error;
    }
  }

  // Async training operations
  async startAsyncTraining(
    studentId: string,
    genuineFiles: File[],
    trainingMode: 'global' = 'global'
  ): Promise<{
    task_id: string;
    message: string;
    student_id: string;
    training_mode: string;
  }> {
    try {
      const formData = new FormData();
      formData.append('student_id', studentId);
      for (const file of genuineFiles) {
        formData.append('genuine_files', file);
      }
      if (trainingMode) {
        formData.append('training_mode', trainingMode);
      }

      const response = await fetch(`${this.baseUrl}/api/training/start-async`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Async training failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Async training error:', error);
      throw error;
    }
  }

  // Verification operations - Owner identification only
  async verifySignature(imageFile: File, studentId?: string): Promise<{
    success: boolean;
    student_id?: string;
    student_name?: string;
    confidence?: number;
    message?: string;
    error?: string;
  }> {
    try {
      const formData = new FormData();
      formData.append('signature_image', imageFile);
      if (studentId) {
        formData.append('student_id', studentId);
      }

      const response = await fetch(`${this.baseUrl}/api/verification/verify`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Verification failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Verification error:', error);
      throw error;
    }
  }

  // Job progress subscription for training
  subscribeToJobProgress(jobId: string, onProgress: (job: {
    job_id: string;
    status: string;
    progress: number;
    current_stage: string;
    estimated_time_remaining?: string;
    start_time?: string;
    end_time?: string;
    error?: string;
    result?: unknown;
    training_metrics: {
      current_epoch: number;
      total_epochs: number;
      accuracy: number;
      val_accuracy: number;
      loss: number;
      val_loss: number;
      precision: number;
      recall: number;
      auc: number;
      val_auc: number;
      learning_rate: number;
      batch_progress: string;
      epoch_progress: string;
    };
  }) => void) {
    const eventSource = new EventSource(`${this.baseUrl}/api/training/job-progress/${jobId}`);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onProgress(data);
        
        // Close connection when job is completed or errored
        if (data.status === 'completed' || data.status === 'error') {
          eventSource.close();
        }
      } catch (error) {
        console.error('Error parsing job progress:', error);
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      eventSource.close();
    };
    
    return eventSource;
  }

  // Get trained models
  async getTrainedModels(): Promise<Array<{
    id: string | number;
    student_name?: string;
    student?: { id?: number; student_id?: string; firstname?: string; surname?: string; full_name?: string };
    student_full_name?: string;
    model_path?: string;
    model?: { path?: string };
    artifact_path?: string;
    training_date?: string;
    created_at?: string;
    accuracy?: number;
  }>> {
    try {
      const response = await fetch(`${this.baseUrl}/api/training/models`);
      if (!response.ok) {
        throw new Error(`Failed to fetch trained models: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching trained models:', error);
      throw error;
    }
  }

  // Get global models
  async getGlobalModels(): Promise<Array<{
    id: string | number;
    student_name?: string;
    student?: { id?: number; student_id?: string; firstname?: string; surname?: string; full_name?: string };
    student_full_name?: string;
    model_path?: string;
    model?: { path?: string };
    artifact_path?: string;
    training_date?: string;
    created_at?: string;
    accuracy?: number;
  }>> {
    try {
      const response = await fetch(`${this.baseUrl}/api/training/global-models`);
      if (!response.ok) {
        throw new Error(`Failed to fetch global models: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching global models:', error);
      throw error;
    }
  }
}

export const aiService = new AIService();
export default aiService;