// AI Service for signature processing and training
// Focus: Owner identification only (no security detection)

const AI_SERVICE_URL = 'http://localhost:8001';

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
    form.append('signature', file);
    const res = await fetch(this.getUrl('/api/signatures/upload'), { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    return data.record as { id:number; student_id:number; label:'genuine'; s3_url:string; s3_key:string };
  }

  async listSignatures(studentId: number) {
    const res = await fetch(this.getUrl(`/api/signatures/list?student_id=${studentId}`));
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'List failed');
    return data.signatures as Array<{ id:number; student_id:number; label:'genuine'; s3_url:string; s3_key:string }>;
  }

  async deleteSignature(signatureId: number, s3Key?: string) {
    const form = new FormData();
    form.append('signature_id', String(signatureId));
    if (s3Key) form.append('s3_key', s3Key);
    const res = await fetch(this.getUrl('/api/signatures/delete'), { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Delete failed');
    return data;
  }

  // Training operations
  async listStudentsWithImages() {
    const res = await fetch(this.getUrl('/api/training/list_students'));
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'List failed');
    return data.items as Array<{ student_id:number; genuine_count?: number; signatures?: Array<{ label:'genuine'; s3_url:string }> }>;
  }

  // GPU-accelerated training with S3 storage
  async startGPUTraining(
    studentId: string,
    genuineFiles: File[],
    useGPU: boolean = true,
    saveToS3: boolean = true
  ): Promise<any> {
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

      const response = await fetch(`${this.baseUrl}/api/training/gpu/start`, {
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

      const response = await fetch(`${this.baseUrl}/api/training/async/start`, {
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

  // Subscribe to async training progress via SSE
  subscribeToJobProgress(taskId: string, onUpdate: (data: any) => void, onError?: (e: any) => void): EventSource {
    const es = new EventSource(this.getUrl(`/api/training/async/progress?task_id=${encodeURIComponent(taskId)}`));
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        onUpdate(data);
      } catch (e) {
        // ignore parse errors
      }
    };
    es.onerror = (e) => {
      if (onError) onError(e);
    };
    return es;
  }

  // Trained models listing (stubs)
  async getTrainedModels(): Promise<any[]> { return []; }
  async getGlobalModels(): Promise<any[]> { return []; }

  // Helper to verify from data URL
  async verifySignatureFromDataURL(dataUrl: string, _sessionId?: number) {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    const file = new File([blob], 'signature.png', { type: blob.type || 'image/png' });
    return this.verifySignature(file);
  }
}

export const aiService = new AIService();
export default aiService;