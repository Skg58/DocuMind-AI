import React from 'react';

const CITATION_REGEX = /\[Page\s+(\d+)\]/gi;

export default function CitationBadge({ children, onClick }) {
  const processNode = (node) => {
    if (typeof node !== 'string') return node;

    const parts = [];
    let lastIndex = 0;
    let match;

    const regex = new RegExp(CITATION_REGEX.source, 'gi');

    while ((match = regex.exec(node)) !== null) {
      if (match.index > lastIndex) {
        parts.push(node.slice(lastIndex, match.index));
      }

      const pageNum = match[1];
      parts.push(
        <button
          key={`citation-${match.index}`}
          type="button"
          className="inline-flex items-center bg-[#f3e7dc] text-[#a85f43] rounded-full
            px-1.5 py-0.5 text-[10px] font-bold cursor-pointer hover:bg-[#ead7c7]
            transition-colors duration-150 mx-0.5"
          title={`Jump to page ${pageNum}`}
          onClick={(e) => {
            e.stopPropagation();
            if (onClick) onClick(Number(pageNum));
          }}
        >
          Page {pageNum}
        </button>
      );

      lastIndex = regex.lastIndex;
    }

    if (lastIndex === 0) return node;
    if (lastIndex < node.length) {
      parts.push(node.slice(lastIndex));
    }

    return parts;
  };

  const processChildren = (children) => {
    if (!children) return children;
    if (typeof children === 'string') return processNode(children);
    if (Array.isArray(children)) {
      return children.map((child, i) => {
        if (typeof child === 'string') {
          const result = processNode(child);
          return Array.isArray(result)
            ? result.map((r, j) => <React.Fragment key={`${i}-${j}`}>{r}</React.Fragment>)
            : result;
        }
        return child;
      });
    }
    return children;
  };

  return <>{processChildren(children)}</>;
}
