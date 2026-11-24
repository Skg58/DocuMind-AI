import React from 'react';
import { BarChart3, MessageSquareText, Upload } from 'lucide-react';

const tabs = [
  { id: 'chat', label: 'Workspace', icon: MessageSquareText },
  { id: 'upload', label: 'Add PDF', icon: Upload },
  { id: 'evaluate', label: 'Quality', icon: BarChart3 },
];

export default function Header({ activeTab, onTabChange, isHealthy, healthInfo }) {
  return (
    <header className="h-[72px] shrink-0 flex items-center justify-between px-5 md:px-7 border-b border-[#dedbd2] bg-[#fbfaf7]/95 backdrop-blur-xl z-30">
      <div className="flex items-center gap-3 min-w-[210px]">
        <div className="relative w-10 h-10 rounded-[13px] bg-[#172c43] shadow-[0_8px_22px_rgba(23,44,67,.18)] flex items-center justify-center">
          <span className="font-serif text-xl italic text-[#f2ae79] leading-none">D</span>
          <span className="absolute -right-1 -bottom-1 w-3 h-3 rounded-full border-2 border-[#fbfaf7] bg-[#d47752]" />
        </div>
        <div>
          <h1 className="font-serif text-[20px] leading-none tracking-[-0.02em] text-[#172c43]">
            DocuMind
          </h1>
          <p className="mt-1 text-[9px] uppercase tracking-[0.24em] font-bold text-[#929187]">
            Reading room
          </p>
        </div>
      </div>

      <nav className="absolute left-1/2 -translate-x-1/2 flex items-center rounded-full border border-[#dedbd2] bg-[#f1efe7] p-1 shadow-sm">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => onTabChange(tab.id)}
              className={`flex items-center gap-2 h-9 px-3.5 rounded-full text-[13px] font-semibold transition-all ${
                active
                  ? 'bg-[#172c43] text-white shadow-[0_5px_14px_rgba(23,44,67,.16)]'
                  : 'text-[#77766e] hover:text-[#172c43]'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="min-w-[210px] flex items-center justify-end gap-2.5">
        <div className="hidden md:flex items-center gap-2 rounded-full border border-[#e2dfd5] bg-white px-3 py-2">
          <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-[#77a281] shadow-[0_0_0_3px_rgba(119,162,129,.12)]' : 'bg-[#c96e66]'}`} />
          <span className="text-[11px] font-bold text-[#65655e]">
            {isHealthy ? 'Engine ready' : 'Reconnecting'}
          </span>
        </div>
        {healthInfo?.version && (
          <span className="hidden xl:inline text-[10px] font-mono text-[#aaa89e]">v{healthInfo.version}</span>
        )}
      </div>
    </header>
  );
}
