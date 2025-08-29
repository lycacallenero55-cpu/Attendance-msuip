import { useState, useCallback } from 'react';

interface UseUnsavedChangesOptions {
  onClose: () => void;
  enabled?: boolean;
}

export function useUnsavedChanges({ onClose, enabled = true }: UseUnsavedChangesOptions) {
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);

  const markAsChanged = useCallback(() => {
    if (enabled) {
      setHasUnsavedChanges(true);
    }
  }, [enabled]);

  const markAsSaved = useCallback(() => {
    setHasUnsavedChanges(false);
  }, []);

  const handleClose = useCallback(() => {
    if (hasUnsavedChanges && enabled) {
      setShowConfirmDialog(true);
    } else {
      onClose();
    }
  }, [hasUnsavedChanges, enabled, onClose]);

  const confirmClose = useCallback(() => {
    setHasUnsavedChanges(false);
    setShowConfirmDialog(false);
    onClose();
  }, [onClose]);

  const cancelClose = useCallback(() => {
    setShowConfirmDialog(false);
  }, []);

  // Prevent dialog from closing when clicking outside
  const handleOpenChange = useCallback((open: boolean) => {
    if (!open) {
      handleClose();
    }
  }, [handleClose]);

  return {
    hasUnsavedChanges,
    showConfirmDialog,
    markAsChanged,
    markAsSaved,
    handleClose,
    confirmClose,
    cancelClose,
    handleOpenChange,
  };
}