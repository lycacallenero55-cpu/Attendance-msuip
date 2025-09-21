export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "12.2.12 (cd3cf9e)"
  }
  public: {
    Tables: {
      admin: {
        Row: {
          approved_at: string | null
          approved_by: string | null
          created_at: string | null
          email: string
          first_name: string
          id: string
          last_name: string
          rejected_at: string | null
          rejected_by: string | null
          status: string | null
          updated_at: string | null
        }
        Insert: {
          approved_at?: string | null
          approved_by?: string | null
          created_at?: string | null
          email: string
          first_name: string
          id?: string
          last_name: string
          rejected_at?: string | null
          rejected_by?: string | null
          status?: string | null
          updated_at?: string | null
        }
        Update: {
          approved_at?: string | null
          approved_by?: string | null
          created_at?: string | null
          email?: string
          first_name?: string
          id?: string
          last_name?: string
          rejected_at?: string | null
          rejected_by?: string | null
          status?: string | null
          updated_at?: string | null
        }
        Relationships: []
      }
      allowed_terms: {
        Row: {
          academic_year: string
          created_at: string
          end_date: string
          id: string
          semester: string
          start_date: string
        }
        Insert: {
          academic_year: string
          created_at?: string
          end_date: string
          id?: string
          semester: string
          start_date: string
        }
        Update: {
          academic_year?: string
          created_at?: string
          end_date?: string
          id?: string
          semester?: string
          start_date?: string
        }
        Relationships: []
      }
      attendance: {
        Row: {
          created_at: string | null
          id: number
          session_id: number
          status: string
          student_id: number
          time_in: string | null
          time_out: string | null
          updated_at: string | null
        }
        Insert: {
          created_at?: string | null
          id?: number
          session_id: number
          status: string
          student_id: number
          time_in?: string | null
          time_out?: string | null
          updated_at?: string | null
        }
        Update: {
          created_at?: string | null
          id?: number
          session_id?: number
          status?: string
          student_id?: number
          time_in?: string | null
          time_out?: string | null
          updated_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "attendance_session_id_fkey"
            columns: ["session_id"]
            isOneToOne: false
            referencedRelation: "sessions"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "attendance_student_id_fkey"
            columns: ["student_id"]
            isOneToOne: false
            referencedRelation: "students"
            referencedColumns: ["id"]
          },
        ]
      }
      excuse_applications: {
        Row: {
          absence_date: string
          created_at: string | null
          documentation_url: string | null
          id: number
          review_notes: string | null
          reviewed_at: string | null
          reviewed_by: string | null
          session_id: number | null
          status: string | null
          student_id: number | null
          updated_at: string | null
        }
        Insert: {
          absence_date: string
          created_at?: string | null
          documentation_url?: string | null
          id?: number
          review_notes?: string | null
          reviewed_at?: string | null
          reviewed_by?: string | null
          session_id?: number | null
          status?: string | null
          student_id?: number | null
          updated_at?: string | null
        }
        Update: {
          absence_date?: string
          created_at?: string | null
          documentation_url?: string | null
          id?: number
          review_notes?: string | null
          reviewed_at?: string | null
          reviewed_by?: string | null
          session_id?: number | null
          status?: string | null
          student_id?: number | null
          updated_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "excuse_applications_session_id_fkey"
            columns: ["session_id"]
            isOneToOne: false
            referencedRelation: "sessions"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "excuse_applications_student_id_fkey"
            columns: ["student_id"]
            isOneToOne: false
            referencedRelation: "students"
            referencedColumns: ["id"]
          },
        ]
      }
      global_trained_models: {
        Row: {
          accuracy: number | null
          centroids_path: string | null
          created_at: string | null
          embedding_spec_path: string | null
          far: number | null
          frr: number | null
          genuine_count: number
          id: number
          is_active: boolean | null
          mappings_path: string | null
          model_path: string
          model_uuid: string
          parent_model_id: number | null
          performance_metrics: Json | null
          s3_key: string | null
          sample_count: number
          status: string
          student_count: number
          training_date: string | null
          training_metrics: Json | null
          updated_at: string | null
          version: number | null
          version_notes: string | null
        }
        Insert: {
          accuracy?: number | null
          centroids_path?: string | null
          created_at?: string | null
          embedding_spec_path?: string | null
          far?: number | null
          frr?: number | null
          genuine_count?: number
          id?: number
          is_active?: boolean | null
          mappings_path?: string | null
          model_path: string
          model_uuid: string
          parent_model_id?: number | null
          performance_metrics?: Json | null
          s3_key?: string | null
          sample_count?: number
          status?: string
          student_count?: number
          training_date?: string | null
          training_metrics?: Json | null
          updated_at?: string | null
          version?: number | null
          version_notes?: string | null
        }
        Update: {
          accuracy?: number | null
          centroids_path?: string | null
          created_at?: string | null
          embedding_spec_path?: string | null
          far?: number | null
          frr?: number | null
          genuine_count?: number
          id?: number
          is_active?: boolean | null
          mappings_path?: string | null
          model_path?: string
          model_uuid?: string
          parent_model_id?: number | null
          performance_metrics?: Json | null
          s3_key?: string | null
          sample_count?: number
          status?: string
          student_count?: number
          training_date?: string | null
          training_metrics?: Json | null
          updated_at?: string | null
          version?: number | null
          version_notes?: string | null
        }
        Relationships: []
      }
      model_ab_tests: {
        Row: {
          created_at: string | null
          created_by: string | null
          description: string | null
          end_date: string | null
          id: number
          is_active: boolean | null
          model_a_id: number
          model_b_id: number
          results: Json | null
          start_date: string | null
          student_id: number
          test_name: string
          traffic_split: number | null
        }
        Insert: {
          created_at?: string | null
          created_by?: string | null
          description?: string | null
          end_date?: string | null
          id?: number
          is_active?: boolean | null
          model_a_id: number
          model_b_id: number
          results?: Json | null
          start_date?: string | null
          student_id: number
          test_name: string
          traffic_split?: number | null
        }
        Update: {
          created_at?: string | null
          created_by?: string | null
          description?: string | null
          end_date?: string | null
          id?: number
          is_active?: boolean | null
          model_a_id?: number
          model_b_id?: number
          results?: Json | null
          start_date?: string | null
          student_id?: number
          test_name?: string
          traffic_split?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "model_ab_tests_model_a_id_fkey"
            columns: ["model_a_id"]
            isOneToOne: false
            referencedRelation: "trained_models"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "model_ab_tests_model_b_id_fkey"
            columns: ["model_b_id"]
            isOneToOne: false
            referencedRelation: "trained_models"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "model_ab_tests_student_id_fkey"
            columns: ["student_id"]
            isOneToOne: false
            referencedRelation: "students"
            referencedColumns: ["id"]
          },
        ]
      }
      model_audit_log: {
        Row: {
          action: string
          id: number
          model_id: number
          new_values: Json | null
          notes: string | null
          old_values: Json | null
          performed_at: string | null
          performed_by: string | null
        }
        Insert: {
          action: string
          id?: number
          model_id: number
          new_values?: Json | null
          notes?: string | null
          old_values?: Json | null
          performed_at?: string | null
          performed_by?: string | null
        }
        Update: {
          action?: string
          id?: number
          model_id?: number
          new_values?: Json | null
          notes?: string | null
          old_values?: Json | null
          performed_at?: string | null
          performed_by?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "model_audit_log_model_id_fkey"
            columns: ["model_id"]
            isOneToOne: false
            referencedRelation: "trained_models"
            referencedColumns: ["id"]
          },
        ]
      }
      model_versions: {
        Row: {
          created_at: string | null
          created_by: string | null
          id: number
          is_active: boolean | null
          model_artifacts: Json | null
          model_id: number
          performance_metrics: Json | null
          version: number
          version_notes: string | null
        }
        Insert: {
          created_at?: string | null
          created_by?: string | null
          id?: number
          is_active?: boolean | null
          model_artifacts?: Json | null
          model_id: number
          performance_metrics?: Json | null
          version: number
          version_notes?: string | null
        }
        Update: {
          created_at?: string | null
          created_by?: string | null
          id?: number
          is_active?: boolean | null
          model_artifacts?: Json | null
          model_id?: number
          performance_metrics?: Json | null
          version?: number
          version_notes?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "model_versions_model_id_fkey"
            columns: ["model_id"]
            isOneToOne: false
            referencedRelation: "trained_models"
            referencedColumns: ["id"]
          },
        ]
      }
      sessions: {
        Row: {
          capacity: string | null
          created_at: string | null
          created_by_user_id: string | null
          date: string
          description: string | null
          id: number
          program: string
          section: string
          time_in: string | null
          time_out: string | null
          title: string
          type: string
          updated_at: string | null
          year: string
        }
        Insert: {
          capacity?: string | null
          created_at?: string | null
          created_by_user_id?: string | null
          date: string
          description?: string | null
          id?: number
          program: string
          section: string
          time_in?: string | null
          time_out?: string | null
          title: string
          type?: string
          updated_at?: string | null
          year: string
        }
        Update: {
          capacity?: string | null
          created_at?: string | null
          created_by_user_id?: string | null
          date?: string
          description?: string | null
          id?: number
          program?: string
          section?: string
          time_in?: string | null
          time_out?: string | null
          title?: string
          type?: string
          updated_at?: string | null
          year?: string
        }
        Relationships: []
      }
      student_signatures: {
        Row: {
          content_hash: string | null
          created_at: string | null
          id: number
          label: string
          s3_key: string
          s3_url: string
          student_id: number
        }
        Insert: {
          content_hash?: string | null
          created_at?: string | null
          id?: never
          label: string
          s3_key: string
          s3_url: string
          student_id: number
        }
        Update: {
          content_hash?: string | null
          created_at?: string | null
          id?: never
          label?: string
          s3_key?: string
          s3_url?: string
          student_id?: number
        }
        Relationships: [
          {
            foreignKeyName: "student_signatures_student_id_fkey"
            columns: ["student_id"]
            isOneToOne: false
            referencedRelation: "students"
            referencedColumns: ["id"]
          },
        ]
      }
      students: {
        Row: {
          created_at: string | null
          firstname: string
          id: number
          middlename: string | null
          program: string
          section: string
          sex: string | null
          student_id: string
          surname: string
          updated_at: string | null
          year: string
        }
        Insert: {
          created_at?: string | null
          firstname: string
          id?: number
          middlename?: string | null
          program: string
          section: string
          sex?: string | null
          student_id: string
          surname: string
          updated_at?: string | null
          year: string
        }
        Update: {
          created_at?: string | null
          firstname?: string
          id?: number
          middlename?: string | null
          program?: string
          section?: string
          sex?: string | null
          student_id?: string
          surname?: string
          updated_at?: string | null
          year?: string
        }
        Relationships: []
      }
      trained_models: {
        Row: {
          accuracy: number | null
          created_at: string | null
          embedding_model_path: string | null
          embedding_s3_key: string | null
          far: number | null
          frr: number | null
          genuine_count: number
          global_model_id: number | null
          id: number
          is_active: boolean | null
          model_path: string
          model_uuid: string | null
          parent_model_id: number | null
          performance_metrics: Json | null
          prototype_centroid: Json | null
          prototype_threshold: number | null
          s3_key: string | null
          sample_count: number
          status: string
          student_id: number
          training_date: string | null
          training_metrics: Json | null
          updated_at: string | null
          version: number | null
          version_notes: string | null
        }
        Insert: {
          accuracy?: number | null
          created_at?: string | null
          embedding_model_path?: string | null
          embedding_s3_key?: string | null
          far?: number | null
          frr?: number | null
          genuine_count?: number
          global_model_id?: number | null
          id?: number
          is_active?: boolean | null
          model_path: string
          model_uuid?: string | null
          parent_model_id?: number | null
          performance_metrics?: Json | null
          prototype_centroid?: Json | null
          prototype_threshold?: number | null
          s3_key?: string | null
          sample_count?: number
          status?: string
          student_id: number
          training_date?: string | null
          training_metrics?: Json | null
          updated_at?: string | null
          version?: number | null
          version_notes?: string | null
        }
        Update: {
          accuracy?: number | null
          created_at?: string | null
          embedding_model_path?: string | null
          embedding_s3_key?: string | null
          far?: number | null
          frr?: number | null
          genuine_count?: number
          global_model_id?: number | null
          id?: number
          is_active?: boolean | null
          model_path?: string
          model_uuid?: string | null
          parent_model_id?: number | null
          performance_metrics?: Json | null
          prototype_centroid?: Json | null
          prototype_threshold?: number | null
          s3_key?: string | null
          sample_count?: number
          status?: string
          student_id?: number
          training_date?: string | null
          training_metrics?: Json | null
          updated_at?: string | null
          version?: number | null
          version_notes?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "trained_models_global_model_id_fkey"
            columns: ["global_model_id"]
            isOneToOne: false
            referencedRelation: "global_trained_models"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "trained_models_student_id_fkey"
            columns: ["student_id"]
            isOneToOne: false
            referencedRelation: "students"
            referencedColumns: ["id"]
          },
        ]
      }
      users: {
        Row: {
          approved_at: string | null
          approved_by: string | null
          created_at: string | null
          email: string
          first_name: string
          id: string
          last_name: string
          rejected_at: string | null
          rejected_by: string | null
          role: string
          status: string | null
          updated_at: string | null
        }
        Insert: {
          approved_at?: string | null
          approved_by?: string | null
          created_at?: string | null
          email: string
          first_name: string
          id?: string
          last_name: string
          rejected_at?: string | null
          rejected_by?: string | null
          role: string
          status?: string | null
          updated_at?: string | null
        }
        Update: {
          approved_at?: string | null
          approved_by?: string | null
          created_at?: string | null
          email?: string
          first_name?: string
          id?: string
          last_name?: string
          rejected_at?: string | null
          rejected_by?: string | null
          role?: string
          status?: string | null
          updated_at?: string | null
        }
        Relationships: []
      }
      verification_results: {
        Row: {
          ab_test_id: number | null
          created_at: string | null
          id: number
          model_id: number
          processing_time_ms: number | null
          student_id: number
          test_signature_path: string | null
          verification_result: Json | null
        }
        Insert: {
          ab_test_id?: number | null
          created_at?: string | null
          id?: number
          model_id: number
          processing_time_ms?: number | null
          student_id: number
          test_signature_path?: string | null
          verification_result?: Json | null
        }
        Update: {
          ab_test_id?: number | null
          created_at?: string | null
          id?: number
          model_id?: number
          processing_time_ms?: number | null
          student_id?: number
          test_signature_path?: string | null
          verification_result?: Json | null
        }
        Relationships: [
          {
            foreignKeyName: "verification_results_ab_test_id_fkey"
            columns: ["ab_test_id"]
            isOneToOne: false
            referencedRelation: "model_ab_tests"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "verification_results_model_id_fkey"
            columns: ["model_id"]
            isOneToOne: false
            referencedRelation: "trained_models"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "verification_results_student_id_fkey"
            columns: ["student_id"]
            isOneToOne: false
            referencedRelation: "students"
            referencedColumns: ["id"]
          },
        ]
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      approve_user: {
        Args: { approver_id: string; user_id: string }
        Returns: Json
      }
      binary_quantize: {
        Args: { "": string } | { "": unknown }
        Returns: unknown
      }
      check_account_exists: {
        Args: { user_id: string }
        Returns: boolean
      }
      check_user_exists: {
        Args: { user_id: string }
        Returns: boolean
      }
      get_role_label: {
        Args: { role_name: string }
        Returns: string
      }
      halfvec_avg: {
        Args: { "": number[] }
        Returns: unknown
      }
      halfvec_out: {
        Args: { "": unknown }
        Returns: unknown
      }
      halfvec_send: {
        Args: { "": unknown }
        Returns: string
      }
      halfvec_typmod_in: {
        Args: { "": unknown[] }
        Returns: number
      }
      hnsw_bit_support: {
        Args: { "": unknown }
        Returns: unknown
      }
      hnsw_halfvec_support: {
        Args: { "": unknown }
        Returns: unknown
      }
      hnsw_sparsevec_support: {
        Args: { "": unknown }
        Returns: unknown
      }
      hnswhandler: {
        Args: { "": unknown }
        Returns: unknown
      }
      is_admin: {
        Args: Record<PropertyKey, never>
        Returns: boolean
      }
      is_staff: {
        Args: Record<PropertyKey, never>
        Returns: boolean
      }
      ivfflat_bit_support: {
        Args: { "": unknown }
        Returns: unknown
      }
      ivfflat_halfvec_support: {
        Args: { "": unknown }
        Returns: unknown
      }
      ivfflathandler: {
        Args: { "": unknown }
        Returns: unknown
      }
      l2_norm: {
        Args: { "": unknown } | { "": unknown }
        Returns: number
      }
      l2_normalize: {
        Args: { "": string } | { "": unknown } | { "": unknown }
        Returns: unknown
      }
      list_students_with_images: {
        Args: Record<PropertyKey, never>
        Returns: Json
      }
      reject_user: {
        Args: { rejector_id: string; user_id: string }
        Returns: Json
      }
      search_similar_signatures: {
        Args: { match_count?: number; query_embedding: string }
        Returns: {
          distance: number
          embedding: string
          student_id: number
        }[]
      }
      sparsevec_out: {
        Args: { "": unknown }
        Returns: unknown
      }
      sparsevec_send: {
        Args: { "": unknown }
        Returns: string
      }
      sparsevec_typmod_in: {
        Args: { "": unknown[] }
        Returns: number
      }
      update_student_signatures: {
        Args: { p_new_signature_url: string; p_student_id: number }
        Returns: Json
      }
      vector_avg: {
        Args: { "": number[] }
        Returns: string
      }
      vector_dims: {
        Args: { "": string } | { "": unknown }
        Returns: number
      }
      vector_norm: {
        Args: { "": string }
        Returns: number
      }
      vector_out: {
        Args: { "": string }
        Returns: unknown
      }
      vector_send: {
        Args: { "": string }
        Returns: string
      }
      vector_typmod_in: {
        Args: { "": unknown[] }
        Returns: number
      }
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
