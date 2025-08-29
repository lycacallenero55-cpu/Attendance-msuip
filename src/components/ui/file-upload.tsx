import * as React from "react"
import { useCallback, useState } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, File, X, Loader2 } from "lucide-react"

import { Button } from "./button"
import { Progress } from "./progress"

type FileWithPreview = File & {
  preview: string
  progress?: number
}

interface FileUploadProps {
  onChange: (files: FileWithPreview[]) => void
  value?: FileWithPreview[]
  maxFiles?: number
  maxSize?: number
  accept?: Record<string, string[]>
  disabled?: boolean
  isUploading?: boolean
}

export function FileUpload({
  onChange,
  value = [],
  maxFiles = 5,
  maxSize = 5 * 1024 * 1024, // 5MB
  accept = {
    'image/*': ['.png', '.jpg', '.jpeg', '.gif'],
    'application/pdf': ['.pdf'],
    'application/msword': ['.doc'],
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  },
  disabled = false,
  isUploading = false,
}: FileUploadProps) {
  const [files, setFiles] = useState<FileWithPreview[]>(value)
  const [rejected, setRejected] = useState<File[]>([])

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: { file: File; errors: { code: string; message: string }[] }[]) => {
      if (disabled) return

      const mappedAccepted = acceptedFiles.map((file) =>
        Object.assign(file, {
          preview: URL.createObjectURL(file),
        })
      )

      setFiles((previousFiles) => {
        const updatedFiles = [...previousFiles, ...mappedAccepted].slice(0, maxFiles)
        onChange(updatedFiles)
        return updatedFiles
      })

      if (rejectedFiles.length > 0) {
        setRejected((previousFiles) => [
          ...previousFiles,
          ...rejectedFiles.map(({ file }) => file),
        ])
      }
    },
    [maxFiles, onChange, disabled]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxSize,
    multiple: maxFiles > 1,
    disabled: disabled || files.length >= maxFiles,
  })

  const removeFile = (name: string) => {
    setFiles((files) => {
      const updatedFiles = files.filter((file) => file.name !== name)
      onChange(updatedFiles)
      return updatedFiles
    })
  }

  const removeAll = () => {
    setFiles([])
    setRejected([])
    onChange([])
  }

  const removeRejected = (name: string) => {
    setRejected((files) => files.filter(({ name: fileName }) => fileName !== name))
  }

  // Clean up object URLs to avoid memory leaks
  React.useEffect(() => {
    return () => {
      files.forEach((file) => URL.revokeObjectURL(file.preview))
    }
  }, [files])

  return (
    <div className="w-full">
      <div
        {...getRootProps({
          className:
            'border-2 border-dashed rounded-lg p-6 text-center transition-colors ' +
            (isDragActive
              ? 'border-primary bg-primary/10'
              : 'border-border bg-background'),
        })}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center justify-center space-y-2">
          <Upload className="h-10 w-10 text-muted-foreground" />
          <div className="text-sm text-muted-foreground">
            {isDragActive ? (
              <p>Drop the files here ...</p>
            ) : (
              <p>
                Drag & drop files here, or click to select files
              </p>
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            {`${Object.values(accept)
              .flat()
              .join(', ')} (max ${maxSize / 1024 / 1024}MB)`}
          </p>
        </div>
      </div>

      {/* Preview */}
      <section className="mt-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-medium">Files to upload</h3>
          <Button
            onClick={removeAll}
            type="button"
            variant="ghost"
            size="sm"
            disabled={files.length === 0 || disabled}
            className="h-8 text-xs text-destructive"
          >
            Remove all
          </Button>
        </div>

        {files.length > 0 && (
          <ul className="mt-2 space-y-2">
            {files.map((file) => (
              <li
                key={file.name}
                className="relative flex items-center justify-between rounded-md border p-2"
              >
                <div className="flex flex-1 items-center space-x-2">
                  <File className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm">{file.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                    {file.progress !== undefined && (
                      <Progress
                        value={file.progress}
                        className="mt-1 h-1.5"
                      />
                    )}
                  </div>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={(e) => {
                    e.stopPropagation()
                    removeFile(file.name)
                  }}
                  disabled={disabled || isUploading}
                >
                  <X className="h-4 w-4" />
                  <span className="sr-only">Remove file</span>
                </Button>
              </li>
            ))}
          </ul>
        )}

        {rejected.length > 0 && (
          <>
            <h3 className="mt-4 text-sm font-medium text-destructive">
              Rejected Files
            </h3>
            <ul className="mt-2 space-y-2">
              {rejected.map(({ name, size }) => (
                <li
                  key={name}
                  className="flex items-center justify-between rounded-md bg-destructive/10 p-2 text-sm"
                >
                  <div className="flex items-center space-x-2">
                    <File className="h-4 w-4" />
                    <div>
                      <p className="font-medium">{name}</p>
                      <p className="text-xs text-muted-foreground">
                        {(size / 1024).toFixed(1)} KB • File type not allowed
                      </p>
                    </div>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-destructive"
                    onClick={() => removeRejected(name)}
                    disabled={disabled}
                  >
                    <X className="h-3.5 w-3.5" />
                    <span className="sr-only">Remove file</span>
                  </Button>
                </li>
              ))}
            </ul>
          </>
        )}

        {isUploading && (
          <div className="mt-4 flex items-center justify-center space-x-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Uploading...</span>
          </div>
        )}
      </section>
    </div>
  )
}
