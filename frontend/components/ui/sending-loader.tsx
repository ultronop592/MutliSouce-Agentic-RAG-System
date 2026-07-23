import React from 'react';

interface TerminalLoaderProps {
  text?: string;
  className?: string;
}

export const Component: React.FC<TerminalLoaderProps> = ({ 
  text = "Connecting...", 
  className = "" 
}) => {
  return (
    <div className={`terminal-loader relative bg-gray-900 border border-gray-700 font-mono text-base p-6 pt-4 w-64 shadow-2xl rounded-lg border-opacity-80 overflow-hidden ${className}`}>
      <div className="terminal-header absolute top-0 left-0 right-0 h-7 bg-gray-800 border-b border-gray-700 rounded-t px-3 flex items-center justify-between">
        <div className="terminal-title text-gray-300 text-xs leading-6 font-semibold flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
          Server Status
        </div>
        <div className="terminal-controls flex gap-1.5">
          <div className="control close w-2.5 h-2.5 rounded-full bg-red-500"></div>
          <div className="control minimize w-2.5 h-2.5 rounded-full bg-yellow-400"></div>
          <div className="control maximize w-2.5 h-2.5 rounded-full bg-green-500"></div>
        </div>
      </div>
      <div className="text text-green-400 inline-block whitespace-nowrap overflow-hidden mt-5 text-sm font-semibold">
        {text}
      </div>
    </div>
  );
};
