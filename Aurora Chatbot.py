import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import customtkinter as ctk
import httpx


# =============================
# Configuration
# =============================
OPENROUTER_API_KEY = "YOUR KEY HERE"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# OpenRouter model id (DeepSeek R1)
# If you have access to a specific variant, you can also try:
#   "deepseek/deepseek-r1:free" or a provider-routed alias.
OPENROUTER_MODEL = "deepseek/deepseek-r1"


# =============================
# API Client (Async, Streaming)
# =============================
class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = OPENROUTER_BASE_URL, timeout_seconds: int = 120):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = httpx.Timeout(timeout_seconds)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def stream_chat(self, messages: List[dict]):
        """
        Sends chat messages and streams the assistant response.

        Yields incremental text tokens via SSE-like stream.
        """
        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "stream": True,
            # You can set additional params like temperature, max_tokens if desired
        }

        # httpx supports streaming via 'stream' context manager
        async with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                try:
                    detail = await response.aread()
                    text = detail.decode("utf-8", errors="ignore")
                except Exception:
                    text = f"HTTP {response.status_code}"
                raise RuntimeError(f"API error: {text} | URL: {self.base_url}")

            # Stream tokens
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: "):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                        # OpenRouter (OpenAI-compatible) delta format
                        delta = payload.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        # Ignore malformed line fragments
                        continue

        return

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# =============================
# Models
# =============================
@dataclass
class ChatMessage:
    role: str  # "user" or "assistant" or "system"
    content: str


@dataclass
class ChatSession:
    title: str = "New Chat"
    messages: List[ChatMessage] = field(default_factory=list)

    def to_openai_messages(self) -> List[dict]:
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]


# =============================
# UI Components
# =============================
class MessageBubble(ctk.CTkFrame):
    def __init__(self, master, text: str, is_user: bool, max_width: int = 720):
        super().__init__(master, corner_radius=16)

        # Colors
        user_bg = ("#7c5cff", "#7c5cff")  # accent purple
        assistant_bg = ("#171920", "#171920")
        text_color = ("#f5f6fb", "#f5f6fb")

        bg_color = user_bg if is_user else assistant_bg
        anchor = "e" if is_user else "w"

        self.configure(fg_color=bg_color)
        self.grid_columnconfigure(0, weight=1)

        self._text_var = ctk.StringVar(value=text)
        self._label = ctk.CTkLabel(
            self,
            textvariable=self._text_var,
            text_color=text_color,
            justify="left",
            wraplength=max_width,
        )
        self._label.grid(row=0, column=0, sticky="w", padx=12, pady=8)

        # Save alignment preference
        self._anchor = anchor

    def update_text(self, extra: str) -> None:
        self._text_var.set(self._text_var.get() + extra)

    @property
    def anchor(self) -> str:
        return self._anchor

    def set_wraplength(self, length: int) -> None:
        try:
            self._label.configure(wraplength=length)
        except Exception:
            pass


class ChatArea(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=("#0f0f0f", "#0f0f0f"))
        self.grid_columnconfigure(0, weight=1)
        self._row = 0
        self._max_bubble_width = 760
        self._bubbles: List[MessageBubble] = []

        # Adjust bubble wrap length responsively
        self.bind("<Configure>", self._on_resize)

    def add_bubble(self, text: str, is_user: bool) -> MessageBubble:
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        bubble = MessageBubble(container, text=text, is_user=is_user, max_width=self._max_bubble_width)
        # Align left or right via internal frame packing
        side = "right" if is_user else "left"
        bubble.pack(side=side, anchor="e" if is_user else "w", padx=12, pady=6)

        container.grid(row=self._row, column=0, sticky="ew", padx=12)
        self._row += 1

        # Auto-scroll to bottom
        self.after(10, self.scroll_to_bottom)
        self._bubbles.append(bubble)
        return bubble

    def scroll_to_bottom(self):
        try:
            self._parent_canvas.yview_moveto(1.0)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _on_resize(self, _event=None):
        # Compute available width inside the scrollable area and set wraplength
        try:
            width = self.winfo_width()
            # Account for paddings and bubble internal padding
            target = max(320, int(width - 48))
            if target != self._max_bubble_width:
                self._max_bubble_width = target
                for b in self._bubbles:
                    b.set_wraplength(self._max_bubble_width)
        except Exception:
            pass


class Sidebar(ctk.CTkFrame):
    def __init__(self, master, on_new_chat: Callable[[], None], on_select_chat: Callable[[int], None]):
        super().__init__(master, corner_radius=0, fg_color=("#0b0c0f", "#0b0c0f"))
        self.on_new_chat = on_new_chat
        self.on_select_chat = on_select_chat

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.new_chat_btn = ctk.CTkButton(self, text="+ New Chat", command=self.on_new_chat, corner_radius=12,
                                          fg_color=("#7c5cff", "#7c5cff"), hover_color=("#6b4fff", "#6b4fff"))
        self.new_chat_btn.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")

        self._separator = ctk.CTkFrame(self, height=1, fg_color=("#15171c", "#15171c"))
        self._separator.grid(row=1, column=0, padx=12, pady=(6, 6), sticky="ew")

        self.history_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.history_frame.grid(row=2, column=0, padx=6, pady=(0, 12), sticky="nsew")
        self.history_buttons: List[ctk.CTkButton] = []

    def refresh_history(self, sessions: List[ChatSession], active_index: int) -> None:
        for btn in self.history_buttons:
            btn.destroy()
        self.history_buttons.clear()

        for idx, sess in enumerate(sessions):
            txt = sess.title if sess.title else f"Chat {idx + 1}"
            btn = ctk.CTkButton(
                self.history_frame,
                text=txt,
                corner_radius=10,
                fg_color=("#101217", "#101217") if idx != active_index else ("#161922", "#161922"),
                hover_color=("#161922", "#161922"),
                command=lambda i=idx: self.on_select_chat(i),
            )
            btn.pack(fill="x", padx=6, pady=4)
            self.history_buttons.append(btn)


class TopBar(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0, fg_color=("#0e0f12", "#0e0f12"))
        self.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(self, text="Aurora Chat ✨", font=ctk.CTkFont(size=18, weight="bold"))
        title.grid(row=0, column=0, padx=16, pady=10, sticky="w")


class InputBar(ctk.CTkFrame):
    def __init__(self, master, on_send: Callable[[str], None]):
        super().__init__(master, corner_radius=0, fg_color=("#0e0f12", "#0e0f12"))
        self.on_send = on_send

        self.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(self, placeholder_text="Type your message...", corner_radius=14)
        self.entry.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="ew")
        self.entry.bind("<Return>", self._on_return)

        self.send_btn = ctk.CTkButton(self, text="Send", corner_radius=14, command=self._on_send_click,
                                      fg_color=("#7c5cff", "#7c5cff"), hover_color=("#6b4fff", "#6b4fff"))
        self.send_btn.grid(row=0, column=1, padx=(6, 12), pady=12)

    def _on_return(self, event):
        self._on_send_click()

    def _on_send_click(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self.on_send(text)


# =============================
# Controller / App
# =============================
class ChatController:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.client = OpenRouterClient(api_key=OPENROUTER_API_KEY)

        # Background asyncio loop in a thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        # Sessions state
        self.sessions: List[ChatSession] = [ChatSession(title="New Chat")]
        self.active_index: int = 0

        # Layout: Sidebar | Main (TopBar, ChatArea, InputBar)
        self._build_ui()
        self._apply_theme()
        self.sidebar.refresh_history(self.sessions, self.active_index)

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _build_ui(self):
        self.root.title("Aurora Chat – AI Assistant")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self.root.grid_columnconfigure(0, weight=0)  # sidebar
        self.root.grid_columnconfigure(1, weight=1)  # main
        self.root.grid_rowconfigure(0, weight=0)     # top bar
        self.root.grid_rowconfigure(1, weight=1)     # chat area
        self.root.grid_rowconfigure(2, weight=0)     # input bar

        self.sidebar = Sidebar(self.root, on_new_chat=self.new_chat, on_select_chat=self.select_chat)
        self.sidebar.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.sidebar.configure(width=240)

        self.topbar = TopBar(self.root)
        self.topbar.grid(row=0, column=1, sticky="ew")

        self.chat_area = ChatArea(self.root)
        self.chat_area.grid(row=1, column=1, sticky="nsew")

        self.input_bar = InputBar(self.root, on_send=self.send_message)
        self.input_bar.grid(row=2, column=1, sticky="ew")

    def _apply_theme(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.root.configure(fg_color=("#07090c", "#07090c"))
        # Subtle shade layering
        self.sidebar.configure(fg_color=("#0b0c0f", "#0b0c0f"))
        self.topbar.configure(fg_color=("#0e0f12", "#0e0f12"))
        self.chat_area.configure(fg_color=("#0a0c10", "#0a0c10"))
        self.input_bar.configure(fg_color=("#0e0f12", "#0e0f12"))

    # ----- Session management -----
    def new_chat(self):
        self.sessions.append(ChatSession(title="New Chat"))
        self.active_index = len(self.sessions) - 1
        self.sidebar.refresh_history(self.sessions, self.active_index)
        self._reload_chat_area()

    def select_chat(self, index: int):
        if 0 <= index < len(self.sessions):
            self.active_index = index
            self.sidebar.refresh_history(self.sessions, self.active_index)
            self._reload_chat_area()

    def _reload_chat_area(self):
        for child in self.chat_area.winfo_children():
            child.destroy()
        self.chat_area._row = 0
        session = self.sessions[self.active_index]
        for msg in session.messages:
            self.chat_area.add_bubble(msg.content, is_user=(msg.role == "user"))
        self.chat_area.after(10, self.chat_area.scroll_to_bottom)

    # ----- Sending and streaming -----
    def send_message(self, text: str):
        session = self.sessions[self.active_index]
        # Add user message
        session.messages.append(ChatMessage(role="user", content=text))
        self.chat_area.add_bubble(text, is_user=True)

        # Add assistant bubble (empty, to be streamed)
        assistant_bubble = self.chat_area.add_bubble("", is_user=False)

        # Kick off async streaming in background loop
        coro = self._stream_and_update_ui(session, assistant_bubble)
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _stream_and_update_ui(self, session: ChatSession, bubble: MessageBubble):
        try:
            # Build messages
            messages = session.to_openai_messages()

            # Ensure session title from first user message
            if session.title == "New Chat":
                for m in session.messages:
                    if m.role == "user" and m.content:
                        session.title = (m.content[:32] + "…") if len(m.content) > 33 else m.content
                        break

            # UI: refresh sidebar title early
            self.root.after(0, lambda: self.sidebar.refresh_history(self.sessions, self.active_index))

            accumulated = []
            async for token in self.client.stream_chat(messages):
                accumulated.append(token)
                # Throttle UI updates slightly for smooth typing effect
                chunk = token
                self.root.after(0, lambda t=chunk: bubble.update_text(t))
                self.root.after(0, self.chat_area.scroll_to_bottom)

            full_text = "".join(accumulated)
            session.messages.append(ChatMessage(role="assistant", content=full_text))
        except Exception as e:
            err_text = f"⚠️ API error: {e}"
            self.root.after(0, lambda: bubble.update_text(err_text))
            session.messages.append(ChatMessage(role="assistant", content=err_text))

    async def close(self):
        await self.client.aclose()
        self._loop.call_soon_threadsafe(self._loop.stop)
        # Give the loop a moment to stop
        time.sleep(0.05)


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    app = ctk.CTk()
    controller = ChatController(app)

    def on_close():
        try:
            asyncio.run(controller.close())
        except Exception:
            pass
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_close)
    app.mainloop()


if __name__ == "__main__":
    main()


