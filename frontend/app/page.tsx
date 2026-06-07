"use client";

import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://mutlisouce-agentic-rag-system-1.onrender.com";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface CollectionInfo {
  name: string;
  points_count: number;
  vectors_count: number;
}

interface TabState {
  messages: Message[];
  input: string;
  isLoading: boolean;
  sessionId: string;
  showUpload: boolean;
  uploadStatus: string | null;
  isUploading: boolean;
}

const TABS_INFO = {
  universal: {
    title: "Universal RAG",
    subtitle: "Search all domains",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2L2 7l10 5 10-5-10-5z" /><path d="M2 17l10 5 10-5" /><path d="M2 12l10 5 10-5" />
      </svg>
    ),
  },
  research_papers: {
    title: "Research Papers",
    subtitle: "Arxiv & Academic",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" />
      </svg>
    ),
  },
  knowledge_base: {
    title: "Knowledge Base",
    subtitle: "General Enterprise Docs",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
      </svg>
    ),
  },
  code_docs: {
    title: "Code Docs",
    subtitle: "APIs & Repositories",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
      </svg>
    ),
  },
  faq_data: {
    title: "FAQ Data",
    subtitle: "Helps & Manuals",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
    ),
  },
};

export default function Home() {
  const [activeTab, setActiveTab] = useState<keyof typeof TABS_INFO>("universal");
  
  // Independent states for each tab
  const [tabStates, setTabStates] = useState<Record<keyof typeof TABS_INFO, TabState>>({
    universal: { messages: [], input: "", isLoading: false, sessionId: crypto.randomUUID(), showUpload: false, uploadStatus: null, isUploading: false },
    research_papers: { messages: [], input: "", isLoading: false, sessionId: crypto.randomUUID(), showUpload: false, uploadStatus: null, isUploading: false },
    knowledge_base: { messages: [], input: "", isLoading: false, sessionId: crypto.randomUUID(), showUpload: false, uploadStatus: null, isUploading: false },
    code_docs: { messages: [], input: "", isLoading: false, sessionId: crypto.randomUUID(), showUpload: false, uploadStatus: null, isUploading: false },
    faq_data: { messages: [], input: "", isLoading: false, sessionId: crypto.randomUUID(), showUpload: false, uploadStatus: null, isUploading: false },
  });

  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [showSidebar, setShowSidebar] = useState(true);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Extract variables for current active tab
  const currentTabState = tabStates[activeTab];
  const activeMessages = currentTabState.messages;
  const activeInput = currentTabState.input;
  const activeIsLoading = currentTabState.isLoading;
  const activeSessionId = currentTabState.sessionId;
  const activeShowUpload = currentTabState.showUpload;
  const activeUploadStatus = currentTabState.uploadStatus;
  const activeIsUploading = currentTabState.isUploading;

  // Update active tab state helper
  const updateActiveTabState = (updates: Partial<TabState>) => {
    setTabStates((prev) => ({
      ...prev,
      [activeTab]: {
        ...prev[activeTab],
        ...updates,
      },
    }));
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [activeMessages]);

  const fetchCollections = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/collections`);
      const data = await res.json();
      setCollections(data.collections || []);
    } catch {
      console.error("Failed to fetch collections");
    }
  }, []);

  useEffect(() => {
    fetchCollections();
  }, [fetchCollections]);

  const handleSend = async () => {
    const textToSend = activeInput.trim();
    if (!textToSend || activeIsLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: textToSend,
      timestamp: new Date(),
    };

    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      timestamp: new Date(),
    };

    // Update active tab state to show user message, empty assistant placeholder, clear input, set loading
    setTabStates((prev) => ({
      ...prev,
      [activeTab]: {
        ...prev[activeTab],
        messages: [...prev[activeTab].messages, userMessage, assistantMessage],
        input: "",
        isLoading: true,
      },
    }));

    try {
      // If we are in a specific tab, target only its collection. Otherwise target all (null)
      const targetCollections = activeTab === "universal" ? null : [activeTab];

      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: textToSend,
          session_id: `${activeTab}-${activeSessionId}`, // Isolate chat memory sessions per tab
          collections: targetCollections,
        }),
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }

      const contentType = res.headers.get("content-type");
      if (contentType?.includes("application/json")) {
        const data = await res.json();
        setTabStates((prev) => ({
          ...prev,
          [activeTab]: {
            ...prev[activeTab],
            messages: prev[activeTab].messages.map((m) =>
              m.id === assistantMessage.id ? { ...m, content: data.answer } : m
            ),
          },
        }));
      } else {
        const reader = res.body?.getReader();
        const decoder = new TextDecoder();

        if (reader) {
          let fullContent = "";
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            fullContent += chunk;
            const currentContent = fullContent;
            setTabStates((prev) => ({
              ...prev,
              [activeTab]: {
                ...prev[activeTab],
                messages: prev[activeTab].messages.map((m) =>
                  m.id === assistantMessage.id ? { ...m, content: currentContent } : m
                ),
              },
            }));
          }
        }
      }
    } catch (error) {
      setTabStates((prev) => ({
        ...prev,
        [activeTab]: {
          ...prev[activeTab],
          messages: prev[activeTab].messages.map((m) =>
            m.id === assistantMessage.id
              ? {
                  ...m,
                  content: `Error: ${error instanceof Error ? error.message : "Failed to connect to the RAG backend. Make sure the server is running on port 8000."}`,
                }
              : m
          ),
        },
      }));
    } finally {
      setTabStates((prev) => ({
        ...prev,
        [activeTab]: {
          ...prev[activeTab],
          isLoading: false,
        },
      }));
    }
  };

  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      updateActiveTabState({ uploadStatus: "Only PDF files are supported" });
      return;
    }

    updateActiveTabState({
      isUploading: true,
      uploadStatus: "Uploading and processing...",
    });

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Upload automatically routes to target collection (defaults to "research_papers" for universal)
      const targetCollection = activeTab === "universal" ? "research_papers" : activeTab;

      const res = await fetch(`${API_BASE}/upload?collection=${targetCollection}`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Upload failed");
      }

      const data = await res.json();
      updateActiveTabState({ uploadStatus: data.message });
      fetchCollections();
    } catch (error) {
      updateActiveTabState({
        uploadStatus: `Upload failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    } finally {
      updateActiveTabState({ isUploading: false });
    }
  };

  const handleClearMemory = async () => {
    try {
      await fetch(`${API_BASE}/memory/${activeTab}-${activeSessionId}`, { method: "DELETE" });
      setTabStates((prev) => ({
        ...prev,
        [activeTab]: {
          ...prev[activeTab],
          messages: [],
        },
      }));
    } catch {
      console.error("Failed to clear memory");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  };

  const getSampleQuestions = (tab: keyof typeof TABS_INFO): string[] => {
    switch (tab) {
      case "research_papers":
        return [
          "Explain how attention works in transformers",
          "What is the architecture of the encoder?",
          "How does multi-head attention improve performance?",
          "Compare self-attention vs recurrence",
        ];
      case "knowledge_base":
        return [
          "What are the onboarding policies for new developers?",
          "Tell me about the corporate security guidelines",
          "What benefits are available to full-time employees?",
          "How do I request time off?",
        ];
      case "code_docs":
        return [
          "Explain the API schema for the /chat route",
          "How does the Orchestrator handle routing?",
          "Where is the pointwise reranking script located?",
          "How do I write a custom agent in the backend?",
        ];
      case "faq_data":
        return [
          "How do I troubleshoot Qdrant connection issues?",
          "What files are supported for ingestion?",
          "How do I clear semantic response cache?",
          "Why am I seeing a green tint in the UI?",
        ];
      default: // universal
        return [
          "What are the latest research papers in attention mechanism?",
          "Summarize our developer coding standard guidelines",
          "Show me the API endpoint mapping for RAG",
          "What should I do if ingestion fails?",
        ];
    }
  };

  // ── Inline markdown renderer (no extra packages) ──────────────────────────
  const renderInline = (text: string): React.ReactNode => {
    const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**') && part.length > 4)
        return <strong key={i} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
      if (part.startsWith('*') && part.endsWith('*') && part.length > 2)
        return <em key={i} className="italic text-white opacity-80">{part.slice(1, -1)}</em>;
      if (part.startsWith('`') && part.endsWith('`') && part.length > 2)
        return <code key={i} className="bg-[var(--color-surface-hover)] border border-[var(--color-border)] px-1.5 py-0.5 rounded text-[0.85em] font-mono text-white">{part.slice(1, -1)}</code>;
      return part;
    });
  };

  const renderMarkdown = (text: string): React.ReactNode => {
    if (!text) return null;
    const lines = text.split('\n');
    const output: React.ReactNode[] = [];
    const listBuf: { type: 'ul' | 'ol'; items: string[] } = { type: 'ul', items: [] };

    const flushList = (key: number) => {
      if (!listBuf.items.length) return;
      const Tag = listBuf.type;
      const cls = Tag === 'ul'
        ? 'list-disc list-outside ml-5 my-2 space-y-1.5 text-white'
        : 'list-decimal list-outside ml-5 my-2 space-y-1.5 text-white';
      output.push(
        <Tag key={`list-${key}`} className={cls}>
          {listBuf.items.map((item, j) => (
            <li key={j} className="text-[15px] leading-relaxed pl-1 text-white">{renderInline(item)}</li>
          ))}
        </Tag>
      );
      listBuf.items = [];
    };

    lines.forEach((line, idx) => {
      // H2 - Main Headings: Dark Green
      if (/^## /.test(line)) {
        flushList(idx);
        output.push(<h2 key={idx} className="text-xl font-bold mt-6 mb-3 text-[var(--color-primary)] pb-1">{renderInline(line.slice(3))}</h2>);
      }
      // H3 - Subheadings: White
      else if (/^### /.test(line)) {
        flushList(idx);
        output.push(<h3 key={idx} className="text-base font-semibold mt-4 mb-2 text-white">{renderInline(line.slice(4))}</h3>);
      }
      // Bullet
      else if (/^[-*] /.test(line)) {
        if (listBuf.type !== 'ul' && listBuf.items.length) flushList(idx);
        listBuf.type = 'ul';
        listBuf.items.push(line.slice(2));
      }
      // Numbered
      else if (/^\d+\. /.test(line)) {
        if (listBuf.type !== 'ol' && listBuf.items.length) flushList(idx);
        listBuf.type = 'ol';
        listBuf.items.push(line.replace(/^\d+\. /, ''));
      }
      // Empty line
      else if (!line.trim()) {
        flushList(idx);
        if (idx > 0 && lines[idx - 1].trim()) output.push(<div key={idx} className="h-3" />);
      }
      // Paragraph
      else {
        flushList(idx);
        output.push(<p key={idx} className="text-[15px] leading-relaxed text-white mb-3">{renderInline(line)}</p>);
      }
    });
    flushList(lines.length);
    return <div className="space-y-1">{output}</div>;
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--color-background)]">
      {/* ═══ Sidebar ═══ */}
      <aside
        className={`${showSidebar ? "w-64" : "w-0"} transition-all duration-300 overflow-hidden flex-shrink-0 bg-[var(--color-surface)] border-r border-[var(--color-border)]`}
      >
        <div className="w-64 h-full flex flex-col">
          {/* Brand */}
          <div className="p-4 border-b border-[var(--color-border)]">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-[var(--color-primary)] flex items-center justify-center flex-shrink-0">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-sm font-semibold text-[var(--color-primary)] leading-tight">RAG Assistant</h1>
                <p className="text-[11px] text-white opacity-70">Enterprise RAG</p>
              </div>
            </div>
          </div>

          {/* RAG Mode Switchers */}
          <div className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
            <div>
              <p className="text-[10px] font-semibold text-[var(--color-text-muted)] uppercase tracking-wider mb-2 px-1">
                RAG Assistants
              </p>
              <div className="space-y-1">
                {Object.entries(TABS_INFO).map(([tabKey, tab]) => {
                  const isActive = activeTab === tabKey;
                  return (
                    <button
                      key={tabKey}
                      onClick={() => setActiveTab(tabKey as any)}
                      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all duration-200 cursor-pointer ${
                        isActive
                          ? "bg-[var(--color-surface-hover)] border border-[var(--color-border)] text-white"
                          : "hover:bg-[var(--color-surface-hover)]/40 border border-transparent text-[var(--color-text-secondary)]"
                      }`}
                    >
                      <div className={`p-1.5 rounded-md flex-shrink-0 transition-colors ${
                        isActive 
                          ? "text-[var(--color-primary)] bg-[var(--color-primary-light)]" 
                          : "text-[var(--color-text-muted)]"
                      }`}>
                        {tab.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className={`text-[13px] font-semibold leading-none ${isActive ? "text-white" : "text-[var(--color-text-secondary)]"}`}>
                          {tab.title}
                        </p>
                        <p className="text-[10px] text-[var(--color-text-muted)] truncate mt-1">
                          {tab.subtitle}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Actions for current active RAG page */}
          <div className="p-3 border-t border-[var(--color-border)] space-y-1.5 bg-black/10">
            <p className="text-[10px] font-semibold text-[var(--color-text-muted)] uppercase tracking-wider mb-1 px-1">
              Actions ({TABS_INFO[activeTab].title})
            </p>
            <button
              onClick={() => {
                setTabStates((prev) => ({
                  ...prev,
                  [activeTab]: {
                    ...prev[activeTab],
                    messages: [],
                  },
                }));
                handleClearMemory();
              }}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-md hover:bg-[var(--color-surface-hover)] transition-colors text-sm font-medium text-white cursor-pointer"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              New Chat
            </button>

            <button
              onClick={() => updateActiveTabState({ showUpload: !activeShowUpload })}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-md transition-colors text-sm font-medium cursor-pointer ${
                activeShowUpload
                  ? "bg-[var(--color-surface-hover)] text-white"
                  : "hover:bg-[var(--color-surface-hover)] text-[var(--color-text-secondary)]"
              }`}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              Upload PDF
            </button>
          </div>

          {/* Upload Zone */}
          {activeShowUpload && (
            <div className="px-3 pb-3 bg-black/10 animate-fade-in">
              <div
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => fileInputRef.current?.click()}
                className="border border-dashed border-[var(--color-border-hover)] rounded-lg p-3 text-center cursor-pointer hover:border-[var(--color-primary)] hover:bg-[var(--color-surface-hover)] transition-colors"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleUpload(file);
                  }}
                />
                {activeIsUploading ? (
                  <div className="flex flex-col items-center gap-2 py-1">
                    <div className="w-5 h-5 border-2 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
                    <p className="text-[10px] text-[var(--color-text-muted)] truncate">Ingesting to {activeTab === "universal" ? "research_papers" : activeTab}...</p>
                  </div>
                ) : (
                  <div className="py-1">
                    <svg className="mx-auto mb-1.5 text-[var(--color-text-muted)]" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="12" y1="18" x2="12" y2="12" /><line x1="9" y1="15" x2="15" y2="15" />
                    </svg>
                    <p className="text-[11px] text-[var(--color-text-secondary)]">Drop PDF or click</p>
                    <p className="text-[9px] text-[var(--color-text-muted)] mt-0.5 leading-none">
                      Routes to {activeTab === "universal" ? "Research Papers" : TABS_INFO[activeTab].title}
                    </p>
                  </div>
                )}
              </div>
              {activeUploadStatus && (
                <div className="mt-2 px-2 py-1.5 rounded bg-[var(--color-surface-hover)] text-[10px] text-center text-white break-words max-h-24 overflow-y-auto">
                  {activeUploadStatus}
                </div>
              )}
            </div>
          )}

          {/* Database Collections Info */}
          <div className="p-3 border-t border-[var(--color-border)] bg-black/20">
            <p className="text-[9px] font-semibold text-[var(--color-text-muted)] uppercase tracking-wider mb-2 px-1">
              Database Ingestion Status
            </p>
            <div className="grid grid-cols-2 gap-1.5">
              {collections.map((col) => (
                <div
                  key={col.name}
                  className="flex flex-col p-1.5 rounded bg-[var(--color-background)] border border-[var(--color-border)] text-left"
                >
                  <span className="text-[10px] text-white truncate capitalize font-medium">
                    {col.name.replace(/_/g, " ")}
                  </span>
                  <span className="text-[9px] text-[var(--color-text-muted)] mt-0.5 font-mono">
                    {col.points_count} points
                  </span>
                </div>
              ))}
              {collections.length === 0 && (
                <p className="col-span-2 text-[10px] text-[var(--color-text-dim)] px-1">
                  Connecting to database...
                </p>
              )}
            </div>
          </div>
        </div>
      </aside>

      {/* ═══ Main Chat Area ═══ */}
      <main className="flex-1 flex flex-col min-w-0 bg-[var(--color-background)]">
        {/* Header */}
        <header className="flex items-center justify-between px-4 py-3 border-b border-[var(--color-border)] bg-[var(--color-surface)]">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="p-1.5 rounded-md hover:bg-[var(--color-surface-hover)] transition-colors text-[var(--color-text-muted)] hover:text-white cursor-pointer"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>
            <div className="flex items-center gap-2">
              <span className="text-[var(--color-primary)]">
                {TABS_INFO[activeTab].icon}
              </span>
              <h2 className="text-sm font-semibold text-[var(--color-primary)]">
                {TABS_INFO[activeTab].title} Assistant
              </h2>
              <span className="text-[10px] text-[var(--color-text-muted)] bg-[var(--color-surface-hover)] px-2 py-0.5 rounded-full border border-[var(--color-border)] uppercase tracking-wider font-semibold">
                {activeTab === "universal" ? "Full Search" : "Filtered RAG"}
              </span>
            </div>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          {activeMessages.length === 0 ? (
            /* ── Empty State ── */
            <div className="flex flex-col items-center justify-center min-h-full text-center max-w-3xl mx-auto px-4 py-12">
              <div className="w-12 h-12 rounded-xl bg-[var(--color-surface)] border border-[var(--color-border)] flex items-center justify-center mb-6 shadow-sm">
                <span className="text-[var(--color-primary)]">
                  {TABS_INFO[activeTab].icon}
                </span>
              </div>
              
              <h2 className="text-2xl font-bold text-[var(--color-primary)] mb-3">
                {activeTab === "universal" 
                  ? "How can I assist your research?" 
                  : `Search within ${TABS_INFO[activeTab].title}`}
              </h2>
              <p className="text-base text-white mb-10 max-w-xl">
                {activeTab === "universal"
                  ? "Ask questions or upload documents. I search across multiple specialized knowledge bases to find precise answers."
                  : `Ask questions about this specific domain. I will only retrieve context from the ${TABS_INFO[activeTab].title} database.`}
              </p>

              {/* Universal Empty State: Show information about all 4 KB categories */}
              {activeTab === "universal" && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 w-full mb-10">
                  <div className="flex flex-col items-center p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-primary)" strokeWidth="2" className="mb-3">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    </svg>
                    <span className="text-[12px] font-semibold text-[var(--color-primary)] uppercase tracking-wide">Research Papers</span>
                    <span className="text-[11px] text-white mt-1 text-center">Academic & Arxiv docs</span>
                  </div>
                  
                  <div className="flex flex-col items-center p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-primary)" strokeWidth="2" className="mb-3">
                      <ellipse cx="12" cy="5" rx="9" ry="3" />
                      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
                      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
                    </svg>
                    <span className="text-[12px] font-semibold text-[var(--color-primary)] uppercase tracking-wide">Knowledge Base</span>
                    <span className="text-[11px] text-white mt-1 text-center">General information</span>
                  </div>

                  <div className="flex flex-col items-center p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-primary)" strokeWidth="2" className="mb-3">
                      <polyline points="16 18 22 12 16 6" />
                      <polyline points="8 6 2 12 8 18" />
                    </svg>
                    <span className="text-[12px] font-semibold text-[var(--color-primary)] uppercase tracking-wide">Code Docs</span>
                    <span className="text-[11px] text-white mt-1 text-center">API & Documentation</span>
                  </div>

                  <div className="flex flex-col items-center p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-primary)" strokeWidth="2" className="mb-3">
                      <circle cx="12" cy="12" r="10" />
                      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                      <line x1="12" y1="17" x2="12.01" y2="17" />
                    </svg>
                    <span className="text-[12px] font-semibold text-[var(--color-primary)] uppercase tracking-wide">FAQ Data</span>
                    <span className="text-[11px] text-white mt-1 text-center">Help & Guides</span>
                  </div>
                </div>
              )}

              {/* Tab-specific details / Upload prompts */}
              {activeTab !== "universal" && (
                <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] max-w-xl w-full mb-10 text-left">
                  <p className="text-[13px] text-white font-semibold mb-1">Targeted Database</p>
                  <p className="text-[12px] text-[var(--color-text-secondary)] leading-relaxed">
                    This assistant is locked to search only the <strong className="text-white capitalize">{activeTab.replace(/_/g, " ")}</strong> collection. You can upload documents directly to this repository using the sidebar button, and queries will bypass generic planning routing.
                  </p>
                </div>
              )}

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl mx-auto">
                {getSampleQuestions(activeTab).map((q) => (
                  <button
                    key={q}
                    onClick={() => {
                      setTabStates((prev) => ({
                        ...prev,
                        [activeTab]: {
                          ...prev[activeTab],
                          input: q,
                        },
                      }));
                      inputRef.current?.focus();
                    }}
                    className="text-left text-[14px] p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)] transition-colors text-white leading-relaxed cursor-pointer"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* ── Chat Messages ── */
            <div className="max-w-3xl mx-auto w-full px-4 py-8 space-y-8">
              {activeMessages.map((msg) => (
                <div key={msg.id} className="flex gap-4 animate-fade-in w-full">
                  {/* Avatar */}
                  <div className="w-8 h-8 rounded-md flex-shrink-0 flex items-center justify-center mt-1">
                    {msg.role === "assistant" ? (
                      <div className="w-full h-full bg-[var(--color-primary)] rounded-md flex items-center justify-center">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                          <path d="M12 2L2 7l10 5 10-5-10-5z" /><path d="M2 17l10 5 10-5" /><path d="M2 12l10 5 10-5" />
                        </svg>
                      </div>
                    ) : (
                      <div className="w-full h-full bg-[var(--color-surface-hover)] rounded-md flex items-center justify-center border border-[var(--color-border)]">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-[var(--color-text-secondary)]">
                          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
                        </svg>
                      </div>
                    )}
                  </div>

                  {/* Message Content */}
                  <div className="flex-1 min-w-0 pt-1.5">
                    <div className="font-medium text-sm text-[var(--color-primary)] mb-1">
                      {msg.role === "assistant" ? `${TABS_INFO[activeTab].title} Assistant` : "You"}
                    </div>
                    <div className={`text-[15px] leading-relaxed text-white`}>
                      {msg.role === "assistant" ? (
                        msg.content ? (
                          renderMarkdown(msg.content)
                        ) : (
                          <div className="flex items-center gap-1.5 h-6">
                            <span className="typing-dot w-1.5 h-1.5 rounded-full bg-[var(--color-text-muted)]" />
                            <span className="typing-dot w-1.5 h-1.5 rounded-full bg-[var(--color-text-muted)]" />
                            <span className="typing-dot w-1.5 h-1.5 rounded-full bg-[var(--color-text-muted)]" />
                          </div>
                        )
                      ) : (
                        msg.content
                      )}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} className="h-4" />
            </div>
          )}
        </div>

        {/* ── Input Area ── */}
        <div className="p-4 mx-auto max-w-3xl w-full">
          <div className="bg-[var(--color-surface)] border border-[var(--color-border)] rounded-xl px-4 py-3 focus-within:border-[var(--color-border-hover)] transition-colors shadow-sm flex items-end gap-3">
            <textarea
              ref={inputRef}
              value={activeInput}
              onChange={(e) => updateActiveTabState({ input: e.target.value })}
              onKeyDown={handleKeyDown}
              placeholder={`Ask ${TABS_INFO[activeTab].title} Assistant...`}
              rows={1}
              className="flex-1 bg-transparent outline-none resize-none text-[15px] text-white placeholder:text-[var(--color-text-muted)] max-h-32 py-0.5"
              style={{
                height: "auto",
                minHeight: "24px",
              }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = "auto";
                target.style.height = Math.min(target.scrollHeight, 128) + "px";
              }}
            />
            <button
              onClick={handleSend}
              disabled={!activeInput.trim() || activeIsLoading}
              className="flex-shrink-0 w-8 h-8 rounded-lg bg-[var(--color-primary)] disabled:bg-[var(--color-surface-hover)] disabled:text-[var(--color-text-muted)] text-white flex items-center justify-center transition-colors disabled:cursor-not-allowed cursor-pointer"
            >
              {activeIsLoading ? (
                <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              )}
            </button>
          </div>
          <p className="text-center text-[11px] text-[var(--color-text-dim)] mt-3">
            AI can make mistakes. Check important info.
          </p>
        </div>
      </main>
    </div>
  );
}
