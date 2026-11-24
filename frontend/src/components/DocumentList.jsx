import React from 'react';
import { FileText, Loader2, Plus, Trash2 } from 'lucide-react';

export default function DocumentList({
  documents,
  selectedDoc,
  onSelectDoc,
  loading,
  onNavigateUpload,
  onDeleteDoc,
  deletingDocId,
  deleteError,
  documentsError,
  onRetryDocuments,
}) {
  return (
    <div className="flex h-full flex-col">
      <div className="px-5 pt-5 pb-3">
        <div className="flex items-end justify-between">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#9b998f]">Library</p>
            <h2 className="mt-1 font-serif text-[21px] text-[#172c43]">Documents</h2>
          </div>
          <button
            type="button"
            onClick={onNavigateUpload}
            className="w-8 h-8 rounded-full border border-[#dedbd2] bg-white text-[#172c43] flex items-center justify-center hover:border-[#c4bca9] transition"
            title="Add PDF"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="mx-5 h-px bg-[#e4e1d7]" />

      <div className="flex-1 overflow-y-auto px-3 py-3">
        {documentsError && (
          <div role="alert" className="mb-3 rounded-xl border border-[#efd5ce] bg-[#fff7f4] p-3">
            <p className="text-[11px] leading-4 text-[#a84f47]">{documentsError}</p>
            <button
              type="button"
              onClick={onRetryDocuments}
              className="mt-2 text-[11px] font-bold text-[#8e4d3e] underline underline-offset-2"
            >
              Try again
            </button>
          </div>
        )}
        {loading && documents.length === 0 ? (
          <div className="py-12 flex justify-center">
            <Loader2 className="w-5 h-5 animate-spin text-[#aaa89e]" />
          </div>
        ) : documents.length === 0 ? (
          <button
            type="button"
            onClick={onNavigateUpload}
            className="group w-full mt-2 rounded-[18px] border border-dashed border-[#d8d3c4] px-4 py-7 text-left hover:border-[#bdae90] hover:bg-white/70 transition"
          >
            <div className="w-10 h-10 rounded-xl bg-[#eeeada] text-[#6f7567] flex items-center justify-center">
              <Plus className="w-4 h-4" />
            </div>
            <p className="mt-4 font-serif text-[17px] text-[#172c43]">Open a document</p>
            <p className="mt-1 text-xs leading-5 text-[#929087]">Add a PDF to begin a private reading session.</p>
          </button>
        ) : (
          <div className="space-y-1.5">
            {documents.map((doc) => {
              const selected = selectedDoc?.doc_id === doc.doc_id;
              const deleting = deletingDocId === doc.doc_id;
              return (
                <div
                  key={doc.doc_id}
                  className={`group relative flex items-center gap-2 rounded-[14px] px-2 py-2 transition ${
                    selected
                      ? 'bg-[#172c43] text-white shadow-[0_8px_22px_rgba(23,44,67,.12)]'
                      : 'text-[#5f6058] hover:bg-white'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => onSelectDoc(doc)}
                    aria-current={selected ? 'true' : undefined}
                    className="flex min-w-0 flex-1 items-center gap-3 rounded-[10px] px-1 py-1 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#d47752]"
                  >
                    <div className={`w-9 h-11 shrink-0 rounded-[7px] border flex items-center justify-center ${
                      selected ? 'border-white/15 bg-white/10' : 'border-[#ddd8c9] bg-[#fbfaf7]'
                    }`}>
                      <FileText className={`w-4 h-4 ${selected ? 'text-[#f2ae79]' : 'text-[#858778]'}`} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-[13px] font-bold">{doc.filename}</p>
                      <p className={`mt-1 text-[10px] ${selected ? 'text-white/55' : 'text-[#9b998f]'}`}>
                        {doc.num_pages ?? '?'} pages · {doc.num_chunks ?? '?'} passages
                      </p>
                    </div>
                  </button>
                  {onDeleteDoc && (
                    <button
                      type="button"
                      title={`Delete ${doc.filename}`}
                      aria-label={`Delete ${doc.filename}`}
                      disabled={deleting}
                      onClick={(event) => {
                        event.stopPropagation();
                        onDeleteDoc(doc);
                      }}
                      className={`shrink-0 rounded-md p-2 transition disabled:cursor-wait disabled:opacity-60 ${
                        selected ? 'text-white/60 hover:text-white hover:bg-white/10' : 'text-[#aaa89e] hover:text-[#b85f56] hover:bg-[#fff3ef]'
                      }`}
                    >
                      {deleting
                        ? <Loader2 className="h-4 w-4 animate-spin" />
                        : <Trash2 className="h-4 w-4" />}
                    </button>
                  )}
                </div>
              );
            })}
            {deleteError && (
              <p role="alert" className="rounded-lg bg-[#fff3ef] px-3 py-2 text-[11px] leading-4 text-[#a84f47]">
                {deleteError.message}
              </p>
            )}
          </div>
        )}
      </div>

      {documents.length > 0 && (
        <div className="px-4 pb-4">
          <div className="rounded-[14px] border border-[#e2dfd5] bg-white/70 px-3 py-2.5">
            <p className="text-[9px] font-bold uppercase tracking-[0.16em] text-[#aaa89e]">Private session</p>
            <p className="mt-1 text-[11px] text-[#77766e]">Files clear after 30 minutes</p>
          </div>
        </div>
      )}
    </div>
  );
}
