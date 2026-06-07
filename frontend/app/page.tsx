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

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(() => crypto.randomUUID());
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, assistantMessage]);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMessage.content,
          session_id: sessionId,
        }),
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }

      const contentType = res.headers.get("content-type");
      if (contentType?.includes("application/json")) {
        const data = await res.json();
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessage.id
              ? { ...m, content: data.answer }
              : m
          )
        );
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
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessage.id
                  ? { ...m, content: currentContent }
                  : m
              )
            );
          }
        }
      }
    } catch (error) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessage.id
            ? {
              ...m,
              content: `Error: ${error instanceof Error ? error.message : "Failed to connect to the RAG backend. Make sure the server is running on port 8000."}`,
            }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadStatus("Only PDF files are supported");
      return;
    }

    setIsUploading(true);
    setUploadStatus("Uploading and processing...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/upload?collection=research_papers`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Upload failed");
      }

      const data = await res.json();
      setUploadStatus(data.message);
      fetchCollections();
    } catch (error) {
      setUploadStatus(
        `Upload failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    } finally {
      setIsUploading(false);
    }
  };

  const handleClearMemory = async () => {
    try {
      await fetch(`${API_BASE}/memory`, { method: "DELETE" });
      setMessages([]);
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
              <div className="w-8 h-8 rounded-lg bg-[var(--color-primary)] flex items-center justify-center">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-sm font-semibold text-[var(--color-primary)]">RAG Assistant</h1>
                <p className="text-[11px] text-white">Enterprise RAG</p>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="p-3 space-y-1.5">
            <button
              onClick={() => {
                setMessages([]);
                handleClearMemory();
              }}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-md hover:bg-[var(--color-surface-hover)] transition-colors text-sm font-medium text-white"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              New Chat
            </button>

            <button
              onClick={() => setShowUpload(!showUpload)}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-md transition-colors text-sm font-medium ${
                showUpload
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
          {showUpload && (
            <div className="px-3 pb-3 animate-fade-in">
              <div
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => fileInputRef.current?.click()}
                className="border border-dashed border-[var(--color-border-hover)] rounded-lg p-4 text-center cursor-pointer hover:border-[var(--color-primary)] hover:bg-[var(--color-surface-hover)] transition-colors"
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
                {isUploading ? (
                  <div className="flex flex-col items-center gap-2 py-1">
                    <div className="w-5 h-5 border-2 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
                    <p className="text-[11px] text-[var(--color-text-muted)]">Processing...</p>
                  </div>
                ) : (
                  <div className="py-1">
                    <svg className="mx-auto mb-1.5 text-[var(--color-text-muted)]" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="12" y1="18" x2="12" y2="12" /><line x1="9" y1="15" x2="15" y2="15" />
                    </svg>
                    <p className="text-[11px] text-[var(--color-text-secondary)]">
                      Drop PDF or click
                    </p>
                  </div>
                )}
              </div>
              {uploadStatus && (
                <div className="mt-2 px-2 py-1.5 rounded-md bg-[var(--color-surface-hover)] text-[10px] text-center text-white break-words">
                  {uploadStatus}
                </div>
              )}
            </div>
          )}

          {/* Collections */}
          <div className="flex-1 overflow-y-auto px-3 pb-4">
            <p className="text-[11px] font-semibold text-[var(--color-text-muted)] uppercase tracking-wider mb-2 px-1 mt-2">
              Knowledge Base
            </p>
            <div className="space-y-0.5">
              {collections.map((col) => (
                <div
                  key={col.name}
                  className="flex items-center justify-between px-2 py-1.5 rounded-md hover:bg-[var(--color-surface-hover)] text-sm transition-colors cursor-default"
                >
                  <span className="text-[13px] text-white truncate capitalize flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-primary)] opacity-80" />
                    {col.name.replace(/_/g, " ")}
                  </span>
                  <span className="text-[10px] text-[var(--color-text-dim)] font-mono">
                    {col.points_count}
                  </span>
                </div>
              ))}
              {collections.length === 0 && (
                <p className="text-[12px] text-[var(--color-text-dim)] px-2 py-1">
                  No collections
                </p>
              )}
            </div>
          </div>
        </div>
      </aside>

      {/* ═══ Main Chat Area ═══ */}
      <main className="flex-1 flex flex-col min-w-0 bg-[var(--color-background)]">
        {/* Header */}
        <header className="flex items-center justify-between px-4 py-3 border-b border-[var(--color-border)]">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="p-1.5 rounded-md hover:bg-[var(--color-surface-hover)] transition-colors text-[var(--color-text-muted)] hover:text-white"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>
            <h2 className="text-sm font-medium text-[var(--color-primary)]">Agentic RAG Assistant</h2>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            /* ── Empty State ── */
            <div className="flex flex-col items-center justify-center h-full text-center max-w-3xl mx-auto px-4">
              <div className="w-12 h-12 rounded-xl bg-[var(--color-surface)] border border-[var(--color-border)] flex items-center justify-center mb-6 shadow-sm">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--color-primary)" strokeWidth="2">
                  <path d="M12 2L2 7l10 5 10-5-10-5z" />
                  <path d="M2 17l10 5 10-5" />
                  <path d="M2 12l10 5 10-5" />
                </svg>
              </div>
              
              <h2 className="text-2xl font-bold text-[var(--color-primary)] mb-3">
                How can I assist your research?
              </h2>
              <p className="text-base text-white mb-10">
                Ask questions or upload documents. I search across multiple specialized knowledge bases to find precise answers.
              </p>

              {/* 4 Types of Knowledge Base Info Section */}
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

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl mx-auto">
                {[
                  "Explain how attention works in transformers",
                  "What is the architecture of the encoder?",
                  "How does multi-head attention improve performance?",
                  "Compare self-attention vs recurrence",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => {
                      setInput(q);
                      inputRef.current?.focus();
                    }}
                    className="text-left text-[14px] p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)] transition-colors text-white leading-relaxed"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* ── Chat Messages ── */
            <div className="max-w-3xl mx-auto w-full px-4 py-8 space-y-8">
              {messages.map((msg) => (
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
                      {msg.role === "assistant" ? "RAG Assistant" : "You"}
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
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question..."
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
              disabled={!input.trim() || isLoading}
              className="flex-shrink-0 w-8 h-8 rounded-lg bg-[var(--color-primary)] disabled:bg-[var(--color-surface-hover)] disabled:text-[var(--color-text-muted)] text-white flex items-center justify-center transition-colors disabled:cursor-not-allowed"
            >
              {isLoading ? (
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
