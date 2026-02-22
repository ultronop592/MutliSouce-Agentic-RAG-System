from collections import defaultdict


class ChatMemory:
    """Simple in-memory chat history, keyed by session_id."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._store: dict[str, list[dict]] = defaultdict(list)

    def update(self, session_id: str, user_msg: str, assistant_msg: str):
        history = self._store[session_id]
        history.append({"user": user_msg, "assistant": assistant_msg})
        # Keep only the last N turns
        if len(history) > self.max_turns:
            self._store[session_id] = history[-self.max_turns :]

    def format_history(self, session_id: str) -> str:
        history = self._store.get(session_id, [])
        if not history:
            return ""
        formatted = ""
        for turn in history:
            formatted += f"User: {turn['user']}\n"
            formatted += f"Assistant: {turn['assistant']}\n\n"
        return formatted

    def clear(self, session_id: str):
        self._store.pop(session_id, None)

    def clear_all(self):
        self._store.clear()


# Global memory instance
chat_memory = ChatMemory()
