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
            referencedRelation: "student_signatures_view"
            referencedColumns: ["student_id"]
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
            referencedRelation: "student_signatures_view"
            referencedColumns: ["student_id"]
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
      signatures: {
        Row: {
          created_at: string
          created_by: string | null
          device_info: Json | null
          features: Json | null
          file_name: string
          file_size: number
          file_type: string
          height: number | null
          id: number
          quality_score: number | null
          storage_path: string
          student_id: number
          updated_at: string
          width: number | null
        }
        Insert: {
          created_at?: string
          created_by?: string | null
          device_info?: Json | null
          features?: Json | null
          file_name: string
          file_size: number
          file_type: string
          height?: number | null
          id?: number
          quality_score?: number | null
          storage_path: string
          student_id: number
          updated_at?: string
          width?: number | null
        }
        Update: {
          created_at?: string
          created_by?: string | null
          device_info?: Json | null
          features?: Json | null
          file_name?: string
          file_size?: number
          file_type?: string
          height?: number | null
          id?: number
          quality_score?: number | null
          storage_path?: string
          student_id?: number
          updated_at?: string
          width?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "signatures_student_id_fkey"
            columns: ["student_id"]
            isOneToOne: false
            referencedRelation: "student_signatures_view"
            referencedColumns: ["student_id"]
          },
          {
            foreignKeyName: "signatures_student_id_fkey"
            columns: ["student_id"]
            isOneToOne: false
            referencedRelation: "students"
            referencedColumns: ["id"]
          },
        ]
      }
      students: {
        Row: {
          address: string | null
          birthday: string | null
          contact_no: string | null
          created_at: string | null
          email: string | null
          firstname: string
          id: number
          middle_initial: string | null
          middlename: string | null
          primary_signature_id: number | null
          program: string
          section: string
          sex: string | null
          signature_url: string | null
          signature_urls: string[] | null
          student_id: string
          surname: string
          updated_at: string | null
          year: string
        }
        Insert: {
          address?: string | null
          birthday?: string | null
          contact_no?: string | null
          created_at?: string | null
          email?: string | null
          firstname: string
          id?: number
          middle_initial?: string | null
          middlename?: string | null
          primary_signature_id?: number | null
          program: string
          section: string
          sex?: string | null
          signature_url?: string | null
          signature_urls?: string[] | null
          student_id: string
          surname: string
          updated_at?: string | null
          year: string
        }
        Update: {
          address?: string | null
          birthday?: string | null
          contact_no?: string | null
          created_at?: string | null
          email?: string | null
          firstname?: string
          id?: number
          middle_initial?: string | null
          middlename?: string | null
          primary_signature_id?: number | null
          program?: string
          section?: string
          sex?: string | null
          signature_url?: string | null
          signature_urls?: string[] | null
          student_id?: string
          surname?: string
          updated_at?: string | null
          year?: string
        }
        Relationships: [
          {
            foreignKeyName: "students_primary_signature_id_fkey"
            columns: ["primary_signature_id"]
            isOneToOne: false
            referencedRelation: "signatures"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "students_primary_signature_id_fkey"
            columns: ["primary_signature_id"]
            isOneToOne: false
            referencedRelation: "student_signatures_view"
            referencedColumns: ["signature_id"]
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
    }
    Views: {
      student_signatures_view: {
        Row: {
          firstname: string | null
          is_primary: boolean | null
          last_updated: string | null
          quality_score: number | null
          signature_date: string | null
          signature_id: number | null
          storage_path: string | null
          student_id: number | null
          student_number: string | null
          surname: string | null
        }
        Relationships: []
      }
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
      compare_signatures: {
        Args: { sig1_id: number; sig2_id: number }
        Returns: number
      }
      get_primary_signature: {
        Args: { student_id_param: number }
        Returns: string
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
