import { openDB, type IDBPDatabase } from "idb";

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
}

export interface ToolCallInfo {
  tool: string;
  input: Record<string, unknown>;
  preview?: string;
  center?: string;
  status: "calling" | "done";
}

export interface DisplayMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCallInfo[];
}

const DB_NAME = "corpus_ui";
const DB_VERSION = 1;

let dbPromise: Promise<IDBPDatabase> | null = null;

function getDB(): Promise<IDBPDatabase> {
  if (!dbPromise) {
    dbPromise = openDB(DB_NAME, DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains("conversations")) {
          db.createObjectStore("conversations", { keyPath: "id" });
        }
        if (!db.objectStoreNames.contains("messages")) {
          db.createObjectStore("messages");
        }
      },
    });
  }
  return dbPromise;
}

/**
 * Initialize storage and migrate from localStorage if needed.
 */
export async function initStorage(): Promise<void> {
  const db = await getDB();

  // Migrate from localStorage on first run
  const legacyConvs = localStorage.getItem("corpus_conversations");
  if (legacyConvs) {
    try {
      const convs: Conversation[] = JSON.parse(legacyConvs);
      const tx = db.transaction(["conversations", "messages"], "readwrite");
      const convStore = tx.objectStore("conversations");
      const msgStore = tx.objectStore("messages");

      for (const conv of convs) {
        await convStore.put(conv);
        const rawMsgs = localStorage.getItem(`corpus_messages_${conv.id}`);
        if (rawMsgs) {
          const msgs: DisplayMessage[] = JSON.parse(rawMsgs);
          await msgStore.put(msgs, conv.id);
        }
      }

      await tx.done;

      // Clean up localStorage
      localStorage.removeItem("corpus_conversations");
      for (const conv of convs) {
        localStorage.removeItem(`corpus_messages_${conv.id}`);
      }
    } catch {
      // If migration fails, leave localStorage intact for retry
    }
  }
}

export async function getConversations(): Promise<Conversation[]> {
  const db = await getDB();
  const all = await db.getAll("conversations");
  return (all as Conversation[]).sort((a, b) => b.createdAt - a.createdAt);
}

export async function saveConversations(convs: Conversation[]): Promise<void> {
  const db = await getDB();
  const tx = db.transaction("conversations", "readwrite");
  const store = tx.objectStore("conversations");
  await store.clear();
  for (const conv of convs) {
    await store.put(conv);
  }
  await tx.done;
}

export async function getMessages(
  conversationId: string
): Promise<DisplayMessage[]> {
  const db = await getDB();
  const msgs = await db.get("messages", conversationId);
  return (msgs as DisplayMessage[] | undefined) ?? [];
}

export async function saveMessages(
  conversationId: string,
  msgs: DisplayMessage[]
): Promise<void> {
  const db = await getDB();
  await db.put("messages", msgs, conversationId);
}

export async function deleteConversation(id: string): Promise<void> {
  const db = await getDB();
  const tx = db.transaction(["conversations", "messages"], "readwrite");
  await tx.objectStore("conversations").delete(id);
  await tx.objectStore("messages").delete(id);
  await tx.done;
}
