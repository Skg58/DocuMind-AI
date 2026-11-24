import React, { useState, useRef, useCallback } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, X, File } from 'lucide-react';

const MAX_FILE_SIZE = 15 * 1024 * 1024; // 15MB

export default function FileUpload({ onUploadSuccess, apiBase, headers = {}, onSessionId }) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [toast, setToast] = useState(null);
  const fileInputRef = useRef(null);

  const showToast = useCallback((message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  }, []);

  const validateFile = useCallback((file) => {
    if (!file) return 'No file selected.';
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      return 'Only PDF files are accepted.';
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File size exceeds 15MB limit. Your file is ${(file.size / (1024 * 1024)).toFixed(1)}MB.`;
    }
    return null;
  }, []);

  const handleFile = useCallback((file) => {
    setResult(null);
    setError(null);
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      showToast(validationError, 'error');
      return;
    }
    setSelectedFile(file);
  }, [validateFile, showToast]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const handleInputChange = useCallback((e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleUpload = useCallback(async () => {
    if (!selectedFile) return;

    setUploading(true);
    setProgress(0);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    let progressInterval;

    try {
      // Simulate progress since fetch doesn't support upload progress natively
      progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + Math.random() * 15;
        });
      }, 300);

      const res = await fetch(`${apiBase}/api/v1/ingest`, {
        method: 'POST',
        body: formData,
        headers,
      });

      setProgress(100);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || errData.message || `Upload failed (${res.status})`);
      }

      const data = await res.json();
      const returnedSessionId = res.headers.get('X-Session-ID') || headers['X-Session-ID'];
      if (returnedSessionId && onSessionId) onSessionId(returnedSessionId);
      setResult(data);
      showToast('Document uploaded and processed successfully!', 'success');
      if (onUploadSuccess) onUploadSuccess(data);
    } catch (err) {
      const msg = err.message || 'Upload failed. Please try again.';
      setError(msg);
      showToast(msg, 'error');
    } finally {
      clearInterval(progressInterval);
      setUploading(false);
    }
  }, [selectedFile, apiBase, onUploadSuccess, showToast]);

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

  return (
    <div className="flex-1 flex items-center justify-center p-6 overflow-auto">
      <div className="w-full max-w-xl space-y-6">
        {/* Title */}
        <div className="text-center">
          <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#aaa89e]">New reading session</p>
          <h2 className="mt-2 font-serif text-[34px] tracking-[-0.03em] text-[#172c43]">Bring a document to the desk</h2>
          <p className="mt-2 text-sm text-[#85867d]">The PDF stays private to this browser session.</p>
        </div>

        {/* Drop Zone */}
        <div
          className={`relative border-2 border-dashed rounded-2xl p-10 text-center transition-all duration-200 cursor-pointer
            ${isDragging
              ? 'border-indigo-500 bg-indigo-500/10'
              : selectedFile
                ? 'border-emerald-500/50 bg-emerald-500/5'
                : 'border-[#dcd7c7] bg-[#f7f5ee] hover:border-[#bdae90] hover:bg-[#f2efe3]'
            }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !uploading && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            className="hidden"
            onChange={handleInputChange}
            disabled={uploading}
          />

          {selectedFile ? (
            <div className="space-y-3">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-emerald-500/10">
                <FileText className="w-8 h-8 text-emerald-400" />
              </div>
              <div>
                <p className="text-[#172c43] font-serif text-lg">{selectedFile.name}</p>
                <p className="text-slate-400 text-sm mt-1">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
              {!uploading && (
                <button
                  onClick={(e) => { e.stopPropagation(); handleReset(); }}
                  className="text-[#aaa89e] hover:text-[#172c43] transition-colors text-sm underline"
                >
                  Choose different file
                </button>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-[#eeeada]">
                <Upload className={`w-8 h-8 ${isDragging ? 'text-[#d47752]' : 'text-[#77766e]'}`} />
              </div>
              <div>
                <p className="text-slate-700 font-medium">
                  {isDragging ? 'Drop your PDF here' : 'Drag & drop your PDF here'}
                </p>
                <p className="text-slate-500 text-sm mt-1">or click to browse · PDF only · 15MB max</p>
              </div>
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {uploading && (
          <div className="space-y-2 animate-fade-in-up">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Processing document...</span>
              <span className="text-indigo-400 font-medium">{Math.round(progress)}%</span>
            </div>
            <div className="h-2 bg-[#e8e5da] rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-indigo-600 to-indigo-400 rounded-full transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Upload Button */}
        {selectedFile && !result && (
          <button
            onClick={handleUpload}
            disabled={uploading}
            className={`w-full py-3.5 px-6 rounded-xl font-semibold text-white transition-all duration-200
              ${uploading
                ? 'bg-indigo-700 cursor-not-allowed opacity-70'
                : 'bg-indigo-600 hover:bg-indigo-500 active:scale-[0.98] shadow-lg shadow-indigo-600/20'
              }`}
          >
            {uploading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Processing...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <Upload className="w-5 h-5" />
                Upload & Process
              </span>
            )}
          </button>
        )}

        {/* Error */}
        {error && (
          <div className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-xl animate-fade-in-up">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-300 font-medium">Upload Failed</p>
              <p className="text-red-400/80 text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Success Result */}
        {result && (
          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4 animate-fade-in-up">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center">
                <CheckCircle className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <p className="text-[#172c43] font-serif text-lg">Document processed</p>
                <p className="text-slate-400 text-sm">{result.message || 'Ready for analysis'}</p>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="bg-[#f7f5ee] border border-[#ebe7dc] rounded-lg p-3 text-center">
                <p className="text-2xl font-serif text-[#172c43]">{result.num_pages ?? '—'}</p>
                <p className="text-slate-500 text-xs mt-1">Pages</p>
              </div>
              <div className="bg-[#f7f5ee] border border-[#ebe7dc] rounded-lg p-3 text-center">
                <p className="text-2xl font-serif text-[#172c43]">{result.num_chunks ?? '—'}</p>
                <p className="text-slate-500 text-xs mt-1">Chunks</p>
              </div>
              <div className="bg-[#f7f5ee] border border-[#ebe7dc] rounded-lg p-3 text-center">
                <p className="text-xs font-mono text-[#77766e] break-all">{result.doc_id ? result.doc_id.slice(0, 8) + '…' : '—'}</p>
                <p className="text-slate-500 text-xs mt-1">Doc ID</p>
              </div>
            </div>

            <button
              onClick={handleReset}
              className="w-full py-2.5 px-4 rounded-lg border border-[#dedbd2] text-[#77766e] hover:border-[#c4bca8] hover:text-[#172c43] transition-colors text-sm font-medium"
            >
              Upload Another Document
            </button>
          </div>
        )}
      </div>

      {/* Toast Notification */}
      {toast && (
        <div className={`fixed bottom-6 right-6 z-50 toast-enter`}>
          <div
            className={`flex items-center gap-3 px-5 py-3.5 rounded-xl shadow-2xl border backdrop-blur-sm
              ${toast.type === 'success'
                ? 'bg-emerald-900/90 border-emerald-700 text-emerald-100'
                : 'bg-red-900/90 border-red-700 text-red-100'
              }`}
          >
            {toast.type === 'success' ? (
              <CheckCircle className="w-5 h-5 text-emerald-400" />
            ) : (
              <AlertCircle className="w-5 h-5 text-red-400" />
            )}
            <p className="text-sm font-medium">{toast.message}</p>
            <button onClick={() => setToast(null)} className="ml-2 opacity-60 hover:opacity-100">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
