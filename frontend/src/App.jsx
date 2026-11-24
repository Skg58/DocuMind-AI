import React, { useState, useEffect, useCallback, useMemo } from 'react';
import Header from './components/Header.jsx';
import ChatInterface from './components/ChatInterface.jsx';
import FileUpload from './components/FileUpload.jsx';
import EvaluationView from './components/EvaluationView.jsx';
import DocumentList from './components/DocumentList.jsx';

const API_BASE = import.meta.env.VITE_API_URL || '';

export default function App() {
  const [sessionId, setSessionId] = useState(() => {
    const saved = localStorage.getItem('documind-session-id');
    if (saved) return saved;
    const created = crypto.randomUUID();
    localStorage.setItem('documind-session-id', created);
    return created;
  });
  const headers = useMemo(() => ({ 'X-Session-ID': sessionId }), [sessionId]);
  const [activeTab, setActiveTab] = useState('chat');
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [messages, setMessages] = useState(() => {
    try {
      return JSON.parse(sessionStorage.getItem('documind-chat-history') || '[]');
    } catch {
      return [];
    }
  });
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [isHealthy, setIsHealthy] = useState(false);
  const [healthInfo, setHealthInfo] = useState(null);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [documentsError, setDocumentsError] = useState(null);
  const [deletingDocId, setDeletingDocId] = useState(null);
  const [deleteError, setDeleteError] = useState(null);

  // Health check
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/health`, { headers });
        if (res.ok) {
          const data = await res.json();
          setIsHealthy(true);
          setHealthInfo(data);
        } else {
          setIsHealthy(false);
        }
      } catch {
        setIsHealthy(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [sessionId]);

  // Fetch documents
  const fetchDocuments = useCallback(async () => {
    setLoadingDocs(true);
    setDocumentsError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/documents`, { headers });
      if (!res.ok) {
        const errorBody = await res.json().catch(() => ({}));
        throw new Error(errorBody.detail || `Could not load documents (${res.status})`);
      }
      const data = await res.json();
      setDocuments(Array.isArray(data) ? data : []);
    } catch (error) {
      setDocumentsError(error.message || 'Could not load documents. Please try again.');
    } finally {
      setLoadingDocs(false);
    }
  }, [sessionId]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  useEffect(() => {
    sessionStorage.setItem('documind-chat-history', JSON.stringify(messages.slice(-20)));
  }, [messages]);

  const handleUploadSuccess = useCallback((result) => {
    if (result.session_id) {
      setSessionId(result.session_id);
      localStorage.setItem('documind-session-id', result.session_id);
    }
    fetchDocuments();
    const newDoc = {
      doc_id: result.doc_id,
      filename: result.filename,
      num_chunks: result.num_chunks,
      num_pages: result.num_pages,
    };
    setSelectedDoc(newDoc);
  }, [fetchDocuments]);

  const handleDeleteDocument = useCallback(async (doc) => {
    if (deletingDocId) return;
    setDeletingDocId(doc.doc_id);
    setDeleteError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(doc.doc_id)}`, {
        method: 'DELETE',
        headers,
      });
      if (!res.ok) {
        const errorBody = await res.json().catch(() => ({}));
        throw new Error(errorBody.detail || `Delete failed (${res.status})`);
      }

      setDocuments((current) => current.filter((item) => item.doc_id !== doc.doc_id));
      setSelectedDoc((current) => current?.doc_id === doc.doc_id ? null : current);
      setMessages((current) => selectedDoc?.doc_id === doc.doc_id ? [] : current);
    } catch (error) {
      setDeleteError({
        docId: doc.doc_id,
        message: error.message || 'Could not remove this document. Please try again.',
      });
    } finally {
      setDeletingDocId(null);
    }
  }, [headers, selectedDoc, deletingDocId]);

  const handleSelectDoc = useCallback((doc) => {
    setSelectedDoc(doc);
    setActiveTab('chat');
  }, []);

  const handleNewChat = useCallback(() => {
    setMessages([]);
    sessionStorage.removeItem('documind-chat-history');
  }, []);

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#f1efe7]">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isHealthy={isHealthy}
        healthInfo={healthInfo}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-[248px] border-r border-[#dedbd2] bg-[#f7f5ee] flex-shrink-0 hidden lg:flex flex-col">
          <DocumentList
            documents={documents}
            selectedDoc={selectedDoc}
            onSelectDoc={handleSelectDoc}
            loading={loadingDocs}
            onNavigateUpload={() => setActiveTab('upload')}
            onDeleteDoc={handleDeleteDocument}
            deletingDocId={deletingDocId}
            deleteError={deleteError}
            documentsError={documentsError}
            onRetryDocuments={fetchDocuments}
          />
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {activeTab === 'chat' && (
            <ChatInterface
              selectedDoc={selectedDoc}
              documents={documents}
              messages={messages}
              setMessages={setMessages}
              onSelectDoc={setSelectedDoc}
              apiBase={API_BASE}
              headers={headers}
              onNewChat={handleNewChat}
              onDeleteDoc={handleDeleteDocument}
              deletingDocId={deletingDocId}
              deleteError={deleteError}
            />
          )}
          {activeTab === 'upload' && (
            <FileUpload
              onUploadSuccess={handleUploadSuccess}
              apiBase={API_BASE}
              headers={headers}
              onSessionId={(id) => {
                setSessionId(id);
                localStorage.setItem('documind-session-id', id);
              }}
            />
          )}
          {activeTab === 'evaluate' && (
            <EvaluationView
              evaluationResults={evaluationResults}
              setEvaluationResults={setEvaluationResults}
              apiBase={API_BASE}
              headers={headers}
            />
          )}
        </main>
      </div>
    </div>
  );
}
