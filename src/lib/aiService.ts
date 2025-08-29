// AI Service Configuration and Client

const AI_BASE_URL = import.meta.env.VITE_AI_BASE_URL || 'http://localhost:8081';

export interface AITrainingResponse {
  success: boolean;
  message: string;
  profile?: {
    student_id: number;
    status: 'untrained' | 'training' | 'ready' | 'error';
    embedding_centroid: number[] | null;
    num_samples: number;
    threshold: number;
    last_trained_at: string | null;
    error_message: string | null;
  };
  error?: string;
}

export interface AIVerificationResponse {
  success: boolean;
  match: boolean;
  predicted_student_id: number | null;
  predicted_student?: {
    id: number;
    student_id: string;
    firstname: string;
    surname: string;
  };
  score: number;
  decision: 'match' | 'no_match' | 'error';
  message: string;
  error?: string;
}

export class AIService {
  private baseUrl: string;

  constructor(baseUrl: string = AI_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Train AI model for a specific student
   */
  async trainStudent(studentId: number): Promise<AITrainingResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/train/${studentId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Training request failed');
      }

      return data;
    } catch (error) {
      console.error('AI training error:', error);
      return {
        success: false,
        message: 'Failed to start training',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Verify signature image
   */
  async verifySignature(
    imageFile: File,
    sessionId?: number
  ): Promise<AIVerificationResponse> {
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      
      if (sessionId) {
        formData.append('session_id', sessionId.toString());
      }

      const response = await fetch(`${this.baseUrl}/verify`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Verification request failed');
      }

      return data;
    } catch (error) {
      console.error('AI verification error:', error);
      return {
        success: false,
        match: false,
        predicted_student_id: null,
        score: 0,
        decision: 'error',
        message: 'Failed to verify signature',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Verify signature from data URL (base64 encoded image)
   */
  async verifySignatureFromDataURL(
    dataURL: string,
    sessionId?: number
  ): Promise<AIVerificationResponse> {
    try {
      // Convert data URL to blob
      const response = await fetch(dataURL);
      const blob = await response.blob();
      
      // Create file from blob
      const file = new File([blob], 'signature.png', { type: 'image/png' });
      
      return this.verifySignature(file, sessionId);
    } catch (error) {
      console.error('Error converting data URL to file:', error);
      return {
        success: false,
        match: false,
        predicted_student_id: null,
        score: 0,
        decision: 'error',
        message: 'Failed to process signature image',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Check AI service health
   */
  async healthCheck(): Promise<{ status: string; healthy: boolean }> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      
      return {
        status: data.status || 'unknown',
        healthy: response.ok && data.status === 'healthy',
      };
    } catch (error) {
      console.error('AI service health check failed:', error);
      return {
        status: 'error',
        healthy: false,
      };
    }
  }
}

// Export a singleton instance
export const aiService = new AIService();

// Export configuration
export const AI_CONFIG = {
  BASE_URL: AI_BASE_URL,
  ENDPOINTS: {
    TRAIN: '/train',
    VERIFY: '/verify',
    HEALTH: '/health',
  },
};