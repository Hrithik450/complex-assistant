from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.base import Checkpoint

class ThreadService(BaseCheckpointSaver):
    def __init__(self, connection):
        self.conn = connection

    def get_tuple(self, thread_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT role, content FROM thread_messages
            WHERE thread_id = %s ORDER BY created_at
        """, (thread_id,))
        rows = cursor.fetchall()
        messages = [{"role": r['role'], "content": r['content']} for r in rows]
        return {"messages": messages}
    
    def put_tuple(self, thread_id, state):
        messages = state["messages"]
        cursor = self.conn.cursor()
        for msg in messages:
            cursor.execute("""
                INSERT INTO thread_messages (thread_id, role, content)
                VALUES (%s, %s, %s)
            """, (thread_id, msg["role"], msg["content"]))
        self.conn.commit()

    def create_new_thread(self, user_id, thread_name, system_prompt):
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO threads (user_id, thread_name) VALUES (%s, %s) RETURNING id
    """, (user_id, thread_name))
        thread_id = cursor.fetchone()['id']

        # Insert the system message as the first message for this thread
        cursor.execute("""
        INSERT INTO thread_messages (thread_id, role, content)
        VALUES (%s, %s, %s)
    """, (thread_id, 'system', system_prompt))
        self.conn.commit()
        return str(thread_id)
    
    def get_last_thread(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute("""
                SELECT id FROM threads
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id,))
        row = cursor.fetchone()
        return row['id'] if row else None