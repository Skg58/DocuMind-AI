import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUp, ChevronDown, FileText, Loader2, Plus, Sparkles, Trash2, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import CitationBadge from './CitationBadge.jsx';

export default function ChatInterface({
  selectedDoc,
  documents,
  messages,
  setMessages,
  onSelectDoc,
  apiBase,
  headers = {},
  onNewChat,
  onDeleteDoc,
  deletingDocId,
  deleteError,
}) {
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [showDocuments, setShowDocuments] = useState(false);
  const [activeCitation, setActiveCitation] = useState(null);
  const endRef = useRef(null);
  const textareaRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = useCallback(async (overrideQuery) => {
    const query = (overrideQuery || input).trim();
    if (!query || !selectedDoc || streaming) return;

    const userMessage = { id: crypto.randomUUID(), role: 'user', content: query, timestamp: Date.now() };
    const assistantMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      sources: [],
      timestamp: Date.now(),
      streaming: true,
    };
    setMessages((current) => [...current, userMessage, assistantMessage]);
    setInput('');
    setStreaming(true);

    try {
      abortRef.current = new AbortController();
      const response = await fetch(`${apiBase}/api/v1/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...headers },
        body: JSON.stringify({
          query,
          doc_id: selectedDoc.doc_id,
          top_k: 5,
          chat_history: messages
            .filter((message) => message.role === 'user' || message.role === 'assistant')
            .slice(-4)
            .map((message) => ({
              role: message.role,
              content: message.content,
            })),
        }),
        signal: abortRef.current.signal,
      });
      if (!response.ok) throw new Error((await response.json()).detail || 'Request failed');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let content = '';
      let sources = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6);
          if (payload === '[DONE]') continue;
          try {
            const event = JSON.parse(payload);
            if (event.error) throw new Error(event.error);
            if (event.sources) {
              sources = event.sources;
              setMessages((current) => {
                const next = [...current];
                next[next.length - 1] = { ...next[next.length - 1], sources };
                return next;
              });
            }
            if (event.token) {
              content += event.token;
              setMessages((current) => {
                const next = [...current];
                next[next.length - 1] = { ...next[next.length - 1], content, sources };
                return next;
              });
            }
          } catch (error) {
            if (error instanceof SyntaxError) {
              content += payload;
            } else {
              throw error;
            }
          }
        }
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        setMessages((current) => {
          const next = [...current];
          next[next.length - 1] = {
            ...next[next.length - 1],
            content: `I couldn't read that request. ${error.message || 'Please try again.'}`,
            streaming: false,
            isError: true,
          };
          return next;
        });
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
      setMessages((current) => {
        const next = [...current];
        if (next[next.length - 1]?.role === 'assistant') {
          next[next.length - 1] = { ...next[next.length - 1], streaming: false };
        }
        return next;
      });
    }
  }, [input, streaming, selectedDoc, apiBase, headers, setMessages, messages]);

  const suggestions = [
    'Summarize the main argument',
    'List the key findings',
    'What are the limitations?',
  ];

  return (
    <div className="relative flex h-full flex-col bg-[#fbfaf7]">
      <div className="h-[54px] shrink-0 px-5 flex items-center justify-between border-b border-[#e4e1d7]">
        <div className="relative">
          <button
            type="button"
            onClick={() => setShowDocuments((current) => !current)}
            className="flex items-center gap-2 rounded-full border border-[#dedbd2] bg-white px-3 py-2 text-[12px] font-bold text-[#5f6058] hover:border-[#c9c1ae]"
          >
            <FileText className="w-3.5 h-3.5 text-[#9b998f]" />
            <span className="max-w-[220px] truncate">
              {selectedDoc?.filename || 'Choose a document'}
            </span>
            <ChevronDown className="w-3.5 h-3.5 text-[#aaa89e]" />
          </button>
          {showDocuments && (
            <div className="absolute left-0 top-[46px] z-40 w-[280px] rounded-[16px] border border-[#dedbd2] bg-white p-1.5 shadow-[0_18px_50px_rgba(40,43,36,.14)]">
              {documents.length === 0 ? (
                <p className="px-3 py-4 text-xs text-[#99978e]">No documents in this session</p>
              ) : documents.map((doc) => (
                <div key={doc.doc_id} className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => {
                      onSelectDoc(doc);
                      setShowDocuments(false);
                    }}
                    className={`min-w-0 flex-1 rounded-[10px] px-3 py-2.5 text-left text-xs transition ${
                      selectedDoc?.doc_id === doc.doc_id
                        ? 'bg-[#eef0e8] text-[#172c43]'
                        : 'text-[#676760] hover:bg-[#f7f5ee]'
                    }`}
                  >
                    <span className="block truncate font-bold">{doc.filename}</span>
                    <span className="mt-1 block text-[10px] text-[#aaa89e]">{doc.num_pages} pages</span>
                  </button>
                  <button
                    type="button"
                    title={`Delete ${doc.filename}`}
                    aria-label={`Delete ${doc.filename}`}
                    disabled={deletingDocId === doc.doc_id}
                    onClick={() => onDeleteDoc?.(doc)}
                    className="rounded-lg p-2 text-[#aaa89e] transition hover:bg-[#fff3ef] hover:text-[#b85f56] disabled:cursor-wait disabled:opacity-50"
                  >
                    {deletingDocId === doc.doc_id
                      ? <Loader2 className="h-4 w-4 animate-spin" />
                      : <Trash2 className="h-4 w-4" />}
                  </button>
                </div>
              ))}
              {deleteError && (
                <p role="alert" className="px-3 py-2 text-[11px] leading-4 text-[#a84f47]">
                  {deleteError.message}
                </p>
              )}
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={onNewChat}
          disabled={!messages.length || streaming}
          title={streaming ? 'Wait for the answer to finish before starting a new chat' : 'Clear chat history'}
          className="flex items-center gap-1.5 rounded-full border border-[#dedbd2] bg-white px-3 py-2 text-[11px] font-bold text-[#77766e] hover:border-[#c4bca8] hover:text-[#172c43] disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Plus className="w-3.5 h-3.5" />
          New chat
        </button>
      </div>

      <div className="chat-scroll flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="mx-auto flex h-full max-w-[720px] flex-col justify-center px-6 py-10">
            <div className="mb-8">
              <div className="mb-5 flex h-11 w-11 items-center justify-center rounded-full border border-[#e0dccd] bg-[#f2efe4] text-[#d47752]">
                <Sparkles className="w-4 h-4" />
              </div>
              <p className="mb-2 text-[10px] font-bold uppercase tracking-[0.22em] text-[#aaa89e]">
                {selectedDoc ? 'Reading session' : 'Start here'}
              </p>
              <h2 className="max-w-[600px] font-serif text-[38px] leading-[1.12] tracking-[-0.035em] text-[#172c43]">
                {selectedDoc
                  ? <>What would you like to uncover in <em>{selectedDoc.filename}</em>?</>
                  : 'Ask your documents, not the internet.'}
              </h2>
            </div>

            {selectedDoc ? (
              <div className="grid gap-2 sm:grid-cols-3">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => send(suggestion)}
                    className="rounded-[14px] border border-[#dedbd2] bg-white px-4 py-3.5 text-left text-[12px] font-semibold text-[#66675f] shadow-[0_4px_16px_rgba(50,51,44,.04)] transition hover:-translate-y-0.5 hover:border-[#c4bca8] hover:text-[#172c43]"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            ) : (
              <button
                type="button"
                onClick={() => documents.length ? setShowDocuments(true) : onSelectDoc?.(null)}
                className="self-start rounded-full bg-[#172c43] px-5 py-3 text-[12px] font-bold text-white shadow-[0_10px_24px_rgba(23,44,67,.14)]"
              >
                {documents.length ? 'Choose a document' : 'Add your first PDF'}
              </button>
            )}
          </div>
        ) : (
          <div className="mx-auto max-w-[760px] space-y-5 px-5 py-8">
            {messages.map((message, index) => (
              <div
                key={message.id || `${message.timestamp || 'saved'}-${message.role}-${index}`}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.role === 'assistant' ? (
                  <article className="w-full max-w-[94%] rounded-[18px] border border-[#e6e1d5] bg-white px-5 py-4 shadow-[0_6px_20px_rgba(54,52,42,.035)] sm:px-6">
                    <div className="mb-3 flex items-center gap-2 border-b border-[#f0ede5] pb-3">
                      <span className="flex h-7 w-7 items-center justify-center rounded-full bg-[#f3eee4] font-serif text-[13px] italic text-[#b86f50]">D</span>
                      <span className="font-serif text-[14px] italic text-[#172c43]">DocuMind</span>
                      <span className="text-[10px] uppercase tracking-[0.12em] text-[#aaa89e]">From your document</span>
                    </div>
                    <div className={`answer-copy border-l-2 pl-4 ${
                      message.isError ? 'border-[#c96e66]' : 'border-[#d47752]'
                    }`}>
                      <ReactMarkdown
                        components={{
                          p: ({ children }) => (
                            <p><CitationBadge onClick={(page) => setActiveCitation({ page, messageId: message.id || message.timestamp, source: message.sources?.find((item) => item.page_num === page) })}>{children}</CitationBadge></p>
                          ),
                          li: ({ children }) => (
                            <li><CitationBadge onClick={(page) => setActiveCitation({ page, messageId: message.id || message.timestamp, source: message.sources?.find((item) => item.page_num === page) })}>{children}</CitationBadge></li>
                          ),
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                      {message.streaming && <span className="stream-caret" />}
                    </div>
                    {message.sources?.length > 0 && !message.streaming && (
                      <div className="mt-4 border-t border-[#f0ede5] pt-3">
                        <p className="mb-2 text-[9px] font-bold uppercase tracking-[0.16em] text-[#aaa89e]">Sources</p>
                        <div className="flex flex-wrap gap-1.5">
                          {message.sources.map((source, sourceIndex) => (
                            <button
                              key={source.chunk_id || sourceIndex}
                              type="button"
                              onClick={() => setActiveCitation({ page: source.page_num, messageId: message.id || message.timestamp, source })}
                              className="flex items-center gap-1.5 rounded-full border border-[#dedbd2] bg-[#fbfaf7] px-2.5 py-1.5 text-[10px] font-bold text-[#77766e] hover:border-[#c4bca8] hover:text-[#172c43] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#d47752]"
                            >
                              <FileText className="w-3 h-3" />
                              Page {source.page_num}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {activeCitation?.messageId === (message.id || message.timestamp) && activeCitation.source && (
                      <div className="mt-3 rounded-[12px] border border-[#ddd7c6] bg-[#f7f5ee] px-4 py-3">
                        <div className="flex items-center justify-between gap-3">
                          <p className="text-[9px] font-bold uppercase tracking-[0.16em] text-[#9b998f]">Source excerpt · Page {activeCitation.page}</p>
                          <button
                            type="button"
                            onClick={() => setActiveCitation(null)}
                            aria-label="Close source excerpt"
                            className="rounded p-1 text-[#929087] hover:bg-white hover:text-[#172c43]"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>
                        <p className="mt-2 text-[12px] leading-5 text-[#5f6058]">{activeCitation.source.text}</p>
                      </div>
                    )}
                  </article>
                ) : (
                  <div className="max-w-[78%]">
                    <p className="mb-1.5 pr-2 text-right text-[9px] font-bold uppercase tracking-[0.14em] text-[#aaa89e]">You</p>
                    <div className="rounded-[16px] rounded-br-[5px] bg-[#172c43] px-4 py-3 text-[13px] leading-5 text-white shadow-[0_8px_20px_rgba(23,44,67,.10)]">
                      {message.content}
                    </div>
                  </div>
                )}
              </div>
            ))}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <div className="shrink-0 px-5 pb-5 pt-3">
        <div className="mx-auto max-w-[760px]">
          <div className="flex items-end gap-2 rounded-[18px] border border-[#ddd8c8] bg-white p-2 shadow-[0_12px_36px_rgba(54,52,42,.08)] focus-within:border-[#b8b09a]">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  send();
                }
              }}
              rows={1}
              disabled={!selectedDoc || streaming}
              placeholder={selectedDoc ? 'Ask a question about this document…' : 'Select a document to begin…'}
              className="min-h-[44px] max-h-28 flex-1 resize-none border-0 bg-transparent px-3 py-[12px] text-[13px] leading-5 text-[#172c43] outline-none placeholder:text-[#aaa89e] disabled:cursor-not-allowed"
            />
            <button
              type="button"
              onClick={() => send()}
              disabled={!selectedDoc || !input.trim() || streaming}
              className="mb-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-[12px] bg-[#d47752] text-white shadow-[0_6px_16px_rgba(212,119,82,.22)] transition hover:bg-[#c66a48] disabled:bg-[#e2dfd5] disabled:text-[#aaa89e] disabled:shadow-none"
            >
              <ArrowUp className="w-4 h-4" />
            </button>
          </div>
          <p className="mt-2 text-center text-[9px] font-semibold uppercase tracking-[0.12em] text-[#b1afa5]">
            Answers cite retrieved passages
          </p>
        </div>
      </div>
    </div>
  );
}
