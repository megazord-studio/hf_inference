import { useState } from 'react';
import { Code, Clipboard, CheckCircle2, ChevronRight } from 'lucide-react';

interface CurlExampleProps {
  command: string;
}

export function CurlExample({ command }: CurlExampleProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(command);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <details className="mt-2 rounded-md border border-base-300 bg-base-100 p-3">
      <summary className="cursor-pointer text-[11px] font-semibold flex items-center gap-1">
        <ChevronRight className="w-3 h-3" /> curl example
      </summary>
      <div className="mt-2 flex items-center justify-between mb-1">
        <span className="font-semibold text-[11px] flex items-center gap-1">
          <Code className="w-3 h-3" /> Copy-ready curl
        </span>
        <button
          type="button"
          className="btn btn-xxs btn-outline gap-1"
          onClick={handleCopy}
        >
          {copied ? <CheckCircle2 className="w-3 h-3" /> : <Clipboard className="w-3 h-3" />}
        </button>
      </div>
      <pre className="text-[10px] whitespace-pre-wrap leading-relaxed max-h-40 overflow-auto">{command}</pre>
    </details>
  );
}

