"use client";

import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8000";

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

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--color-background)]">

      {/* ═══ Sidebar ═══ */}
      <aside
        className={`${showSidebar ? "w-72" : "w-0"} transition-all duration-300 overflow-hidden flex-shrink-0`}
      >
        <div className="w-72 h-full flex flex-col bg-[var(--color-surface)] border-r border-[var(--color-border)]"
          style={{ boxShadow: "var(--shadow-sm)" }}>

          {/* Brand */}
          <div className="p-5 border-b border-[var(--color-border)]">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[var(--color-primary)] flex items-center justify-center"
                style={{ boxShadow: "0 2px 8px rgba(13, 148, 136, 0.2)" }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-sm font-semibold text-[var(--color-text)]">RAG Assistant</h1>
                <p className="text-xs text-[var(--color-text-muted)]">Multi-Source AI</p>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="p-4 space-y-1.5">
            <button
              onClick={() => {
                setMessages([]);
                handleClearMemory();
              }}
              className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)] transition-all text-sm font-medium text-[var(--color-text)]"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              New Chat
            </button>

            <button
              onClick={() => setShowUpload(!showUpload)}
              className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg hover:bg-[var(--color-surface-hover)] transition-all text-sm text-[var(--color-text-secondary)]"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              Upload PDF
            </button>
          </div>

          {/* Upload Zone */}
          {showUpload && (
            <div className="px-4 pb-4 animate-fade-in">
              <div
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-[var(--color-border)] rounded-xl p-6 text-center cursor-pointer hover:border-[var(--color-primary)] hover:bg-[var(--color-primary-light)] transition-all"
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
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-8 h-8 border-2 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
                    <p className="text-xs text-[var(--color-text-muted)]">Processing...</p>
                  </div>
                ) : (
                  <>
                    <svg className="mx-auto mb-2 text-[var(--color-text-dim)]" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="12" y1="18" x2="12" y2="12" /><line x1="9" y1="15" x2="15" y2="15" />
                    </svg>
                    <p className="text-xs text-[var(--color-text-muted)]">
                      Drop PDF here or click to browse
                    </p>
                  </>
                )}
              </div>
              {uploadStatus && (
                <div className="mt-2 px-3 py-2 rounded-lg bg-[var(--color-surface-hover)] text-xs text-center text-[var(--color-text-secondary)]">
                  {uploadStatus}
                </div>
              )}
            </div>
          )}

          {/* Collections */}
          <div className="flex-1 overflow-y-auto px-4 pb-4">
            <p className="text-xs font-semibold text-[var(--color-text-dim)] uppercase tracking-wider mb-3">
              Knowledge Sources
            </p>
            <div className="space-y-1">
              {collections.map((col) => (
                <div
                  key={col.name}
                  className="flex items-center justify-between px-3 py-2.5 rounded-lg bg-[var(--color-background)] text-sm transition-colors hover:bg-[var(--color-surface-hover)]"
                >
                  <span className="text-[var(--color-text-secondary)] truncate capitalize">
                    {col.name.replace(/_/g, " ")}
                  </span>
                  <span
                    className={`text-xs font-mono px-1.5 py-0.5 rounded ${col.points_count > 0
                      ? "text-[var(--color-success)] bg-emerald-50"
                      : "text-[var(--color-text-dim)] bg-gray-50"
                      }`}
                  >
                    {col.points_count}
                  </span>
                </div>
              ))}
              {collections.length === 0 && (
                <p className="text-xs text-[var(--color-text-dim)] italic px-3 py-2">
                  Loading collections...
                </p>
              )}
            </div>
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-[var(--color-border)]">
            <div className="flex items-center gap-2 text-xs text-[var(--color-text-dim)]">
              <div className="w-2 h-2 rounded-full bg-[var(--color-success)] animate-subtle-pulse" />
              Gemini 2.5 Flash · Qdrant
            </div>
          </div>
        </div>
      </aside>

      {/* ═══ Main Chat Area ═══ */}
      <main className="flex-1 flex flex-col min-w-0">

        {/* Header */}
        <header className="flex items-center gap-3 px-5 py-3.5 border-b border-[var(--color-border)] bg-[var(--color-surface)]"
          style={{ boxShadow: "var(--shadow-sm)" }}>
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className="p-2 rounded-lg hover:bg-[var(--color-surface-hover)] transition-colors"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
          <div className="flex-1">
            <h2 className="text-sm font-semibold text-[var(--color-text)]">Multi-Source RAG Chat</h2>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-[var(--color-text-dim)] bg-[var(--color-background)] px-2.5 py-1 rounded-full">
              {messages.filter((m) => m.role === "user").length} messages
            </span>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          {messages.length === 0 ? (
            /* ── Empty State ── */
            <div className="flex flex-col items-center justify-center h-full text-center max-w-lg mx-auto">
              <div className="w-16 h-16 rounded-2xl bg-[var(--color-primary)] flex items-center justify-center mb-6"
                style={{ boxShadow: "0 4px 16px rgba(13, 148, 136, 0.15)" }}>
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5">
                  <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-[var(--color-text)] mb-2">
                Multi-Source RAG Assistant
              </h2>
              <p className="text-sm text-[var(--color-text-muted)] mb-8 leading-relaxed">
                Upload PDFs and ask questions. I&apos;ll search across your knowledge sources to find the best answers.
              </p>
              <div className="grid grid-cols-2 gap-3 w-full">
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
                    className="text-left text-xs p-3.5 rounded-xl border border-[var(--color-border)] hover:border-[var(--color-primary)] hover:bg-[var(--color-primary-light)] transition-all text-[var(--color-text-secondary)] leading-relaxed"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* ── Chat Messages ── */
            <div className="max-w-3xl mx-auto space-y-5">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex gap-3 animate-fade-in ${msg.role === "user" ? "justify-end" : "justify-start"
                    }`}
                >
                  {/* Assistant Avatar */}
                  {msg.role === "assistant" && (
                    <div className="w-8 h-8 rounded-lg bg-[var(--color-primary)] flex items-center justify-center flex-shrink-0 mt-0.5"
                      style={{ boxShadow: "0 1px 4px rgba(13, 148, 136, 0.15)" }}>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                        <path d="M12 2L2 7l10 5 10-5-10-5z" /><path d="M2 17l10 5 10-5" /><path d="M2 12l10 5 10-5" />
                      </svg>
                    </div>
                  )}

                  {/* Message Bubble */}
                  <div
                    className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${msg.role === "user"
                      ? "bg-[var(--color-primary)] text-white rounded-br-md"
                      : "bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)] rounded-bl-md"
                      }`}
                    style={msg.role === "assistant" ? { boxShadow: "var(--shadow-sm)" } : { boxShadow: "0 1px 4px rgba(13, 148, 136, 0.15)" }}
                  >
                    {msg.content || (
                      <div className="flex items-center gap-1.5 py-1">
                        <span className="typing-dot w-1.5 h-1.5 rounded-full bg-[var(--color-text-muted)]" />
                        <span className="typing-dot w-1.5 h-1.5 rounded-full bg-[var(--color-text-muted)]" />
                        <span className="typing-dot w-1.5 h-1.5 rounded-full bg-[var(--color-text-muted)]" />
                      </div>
                    )}
                  </div>

                  {/* User Avatar */}
                  {msg.role === "user" && (
                    <div className="w-8 h-8 rounded-lg bg-[var(--color-surface-active)] flex items-center justify-center flex-shrink-0 mt-0.5">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--color-text-muted)" strokeWidth="2">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
                      </svg>
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* ── Input Area ── */}
        <div className="p-4 border-t border-[var(--color-border)] bg-[var(--color-surface)]">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end gap-3 bg-[var(--color-background)] border border-[var(--color-border)] rounded-2xl px-4 py-3 focus-within:border-[var(--color-primary)] transition-colors"
              style={{ boxShadow: "var(--shadow-sm)" }}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about your documents..."
                rows={1}
                className="flex-1 bg-transparent outline-none resize-none text-sm text-[var(--color-text)] placeholder:text-[var(--color-text-dim)] max-h-32"
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
                className="flex-shrink-0 w-9 h-9 rounded-xl bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-all"
              >
                {isLoading ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                )}
              </button>
            </div>
            <p className="text-center text-xs text-[var(--color-text-dim)] mt-2.5">
              Powered by Gemini 2.5 Flash · Hybrid Vector + BM25 Retrieval · Qdrant Cloud
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
