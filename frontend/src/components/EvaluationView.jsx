import React, { useState, useCallback } from 'react';
import { Play, Loader2, BarChart3, ChevronDown, ChevronRight, Target, Crosshair, ShieldCheck, Star } from 'lucide-react';

function MetricCard({ label, value, icon: Icon, color }) {
  const percentage = Math.round((value ?? 0) * 100);
  const colorMap = {
    indigo: { bg: 'bg-[#eef0e8]', text: 'text-[#172c43]', bar: 'bg-[#172c43]' },
    emerald: { bg: 'bg-[#edf3ed]', text: 'text-[#64816b]', bar: 'bg-[#77a281]' },
    amber: { bg: 'bg-[#f7eee4]', text: 'text-[#b86f50]', bar: 'bg-[#d47752]' },
    blue: { bg: 'bg-[#edf1f3]', text: 'text-[#5f7380]', bar: 'bg-[#718896]' },
  };
  const c = colorMap[color] || colorMap.indigo;

  return (
    <div className="rounded-[18px] border border-[#dedbd2] bg-white p-5 shadow-[0_8px_24px_rgba(50,51,44,.04)]">
      <div className="flex items-center gap-3 mb-4">
        <div className={`w-10 h-10 rounded-lg ${c.bg} flex items-center justify-center`}>
          <Icon className={`w-5 h-5 ${c.text}`} />
        </div>
        <div>
          <p className="text-sm text-slate-400">{label}</p>
          <p className={`text-2xl font-bold ${c.text}`}>
            {value !== null && value !== undefined ? `${percentage}%` : '—'}
          </p>
        </div>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full ${c.bar} rounded-full transition-all duration-1000 ease-out`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default function EvaluationView({ evaluationResults, setEvaluationResults, apiBase, headers = {} }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedRows, setExpandedRows] = useState({});

  const runEvaluation = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/api/v1/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...headers },
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Evaluation failed (${res.status})`);
      }
      const data = await res.json();
      setEvaluationResults(data);
    } catch (err) {
      setError(err.message || 'Evaluation failed. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [apiBase, headers, setEvaluationResults]);

  const toggleRow = useCallback((idx) => {
    setExpandedRows((prev) => ({ ...prev, [idx]: !prev[idx] }));
  }, []);

  const metrics = evaluationResults?.metrics || {};
  const perQueryResults = evaluationResults?.per_query_results || [];
  const summary = evaluationResults?.summary || '';

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-5xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#aaa89e]">Quality desk</p>
            <h2 className="mt-1 font-serif text-[32px] tracking-[-0.03em] text-[#172c43]">How well does it read?</h2>
            <p className="mt-1 text-sm text-[#85867d]">Measure retrieval and answer quality against the sample set.</p>
          </div>
          <button
            onClick={runEvaluation}
            disabled={loading}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-white transition-all duration-200
              ${loading
                ? 'bg-indigo-700 cursor-not-allowed opacity-70'
                : 'bg-[#172c43] hover:bg-[#263f59] active:scale-[0.98] shadow-lg shadow-[#172c43]/15'
              }`}
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Run Evaluation
              </>
            )}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div role="alert" className="p-4 bg-[#fff3ef] border border-[#efd5ce] rounded-xl text-[#a84f47] text-sm animate-fade-in-up">
            {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20 space-y-4">
            <div className="relative">
              <div className="w-16 h-16 rounded-full border-4 border-slate-700 border-t-indigo-500 animate-spin" />
            </div>
            <p className="text-slate-400">Running evaluation pipeline...</p>
            <p className="text-slate-600 text-sm">This may take a minute</p>
          </div>
        )}

        {/* Metrics Grid */}
        {!loading && evaluationResults && (
          <div className="space-y-8 animate-fade-in-up">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                label="Precision@K"
                value={metrics.avg_precision_at_k ?? metrics.precision_at_k ?? metrics['precision@k'] ?? metrics.precision}
                icon={Target}
                color="indigo"
              />
              <MetricCard
                label="MRR"
                value={metrics.avg_mrr ?? metrics.mrr ?? metrics.mean_reciprocal_rank}
                icon={Crosshair}
                color="blue"
              />
              <MetricCard
                label="Faithfulness"
                value={metrics.avg_faithfulness ?? metrics.faithfulness ?? metrics.faithfulness_score}
                icon={ShieldCheck}
                color="emerald"
              />
              <MetricCard
                label="Relevance"
                value={metrics.avg_relevance ?? metrics.relevance ?? metrics.relevance_score ?? metrics.answer_relevance}
                icon={Star}
                color="amber"
              />
            </div>

            {/* Summary */}
            {summary && (
              <div className="bg-white border border-slate-200 rounded-xl p-5">
                <h3 className="text-sm font-semibold text-[#172c43] mb-3 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-indigo-400" />
                  Summary
                </h3>
                <p className="text-slate-400 text-sm leading-relaxed whitespace-pre-wrap">{summary}</p>
              </div>
            )}

            {/* Per-Query Results */}
            {perQueryResults.length > 0 && (
              <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
                <div className="px-5 py-4 border-b border-slate-100">
                  <h3 className="text-sm font-semibold text-[#172c43] flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-indigo-400" />
                    Per-Query Results ({perQueryResults.length})
                  </h3>
                </div>

                <div className="divide-y divide-[#eeeae0]">
                  {perQueryResults.map((item, idx) => (
                    <div key={idx}>
                      <button
                        onClick={() => toggleRow(idx)}
                        className="w-full px-5 py-3.5 flex items-center gap-3 hover:bg-slate-50 transition-colors text-left"
                      >
                        {expandedRows[idx] ? (
                          <ChevronDown className="w-4 h-4 text-slate-500 flex-shrink-0" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-slate-500 flex-shrink-0" />
                        )}
                        <span className="text-sm text-slate-700 flex-1 truncate">
                          {item.query || `Query ${idx + 1}`}
                        </span>
                        {item.score !== undefined && (
                          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                            item.score >= 0.7
                              ? 'bg-emerald-500/10 text-emerald-400'
                              : item.score >= 0.4
                                ? 'bg-amber-500/10 text-amber-400'
                                : 'bg-red-500/10 text-red-400'
                          }`}>
                            {(item.score * 100).toFixed(0)}%
                          </span>
                        )}
                      </button>

                      {expandedRows[idx] && (
                        <div className="px-5 pb-4 pl-12 space-y-3 animate-fade-in-up">
                          {item.answer && (
                            <div>
                              <p className="text-xs text-slate-500 mb-1">Answer</p>
                              <p className="text-sm text-[#5f6058]">{item.answer}</p>
                            </div>
                          )}
                          {item.expected && (
                            <div>
                              <p className="text-xs text-slate-500 mb-1">Expected</p>
                              <p className="text-sm text-slate-400">{item.expected}</p>
                            </div>
                          )}
                          {item.sources && item.sources.length > 0 && (
                            <div>
                              <p className="text-xs text-slate-500 mb-1">Sources</p>
                              <div className="flex flex-wrap gap-1">
                                {item.sources.map((s, sIdx) => (
                                  <span key={sIdx} className="text-xs bg-[#f2efe4] text-[#77766e] px-2 py-1 rounded">
                                    {s.page !== undefined ? `Page ${s.page}` : `Source ${sIdx + 1}`}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          {/* Show individual metrics if available */}
                          <div className="flex gap-4 text-xs">
                            {item.faithfulness !== undefined && (
                              <span className="text-slate-500">
                                Faithfulness: <span className="text-slate-300">{(item.faithfulness * 100).toFixed(0)}%</span>
                              </span>
                            )}
                            {item.relevance !== undefined && (
                              <span className="text-slate-500">
                                Relevance: <span className="text-slate-300">{(item.relevance * 100).toFixed(0)}%</span>
                              </span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Empty State */}
        {!loading && !evaluationResults && !error && (
          <div className="flex flex-col items-center justify-center py-20 text-center space-y-4 opacity-60">
            <div className="w-20 h-20 rounded-2xl bg-indigo-50 flex items-center justify-center">
              <BarChart3 className="w-10 h-10 text-indigo-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-800">No Evaluation Results</h3>
              <p className="text-slate-500 text-sm mt-1 max-w-md">
                Click "Run Evaluation" to test your RAG pipeline's retrieval and generation quality.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
