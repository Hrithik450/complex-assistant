import json

class ThreadService():
    def __init__(self, connection, redis_client):
        self.conn = connection
        self.redis = redis_client

    # We now fetch the creation date to show in the sidebar.
    def get_all_threads_for_user(self, user_id):
        """Retrieves all threads for a given user, ordered by most recent."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, title, created_at 
            FROM thread
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """, (user_id,))
        rows = cursor.fetchall()
        threads = [{"id": str(r['id']), "title": r['title'], "created_at": r['created_at']} for r in rows]
        return threads
    
    # --- NEW FUNCTION ---
    def rename_thread(self, thread_id, new_title):
        """Renames a specific thread."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE thread SET title = %s WHERE id = %s
        """, (new_title, thread_id))
        self.conn.commit()
        # Invalidate any caches related to this thread if necessary
        # For now, we assume the title isn't heavily cached elsewhere

    # --- NEW FUNCTION ---
    def delete_thread(self, thread_id):
        """Deletes a specific thread and all its messages."""
        cursor = self.conn.cursor()
        # The ON DELETE CASCADE in your database schema will handle deleting messages
        cursor.execute("""
            DELETE FROM thread WHERE id = %s
        """, (thread_id,))
        self.conn.commit()
        # Invalidate any caches related to this thread
        cache_key = f"thread:{thread_id}:thread_messages"
        self.redis.delete(cache_key)

    def get_thread_messages(self, thread_id):
        # Check Redis cache first
        cache_key = f"thread:{thread_id}:thread_messages"
        cached_value = self.redis.get(cache_key)
        if cached_value:
            return json.loads(cached_value)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT role, content 
            FROM thread_messages
            WHERE thread_id = %s 
            ORDER BY created_at DESC
            LIMIT 5
        """, (thread_id,))
        rows = cursor.fetchall()

        # Convert rows into list of dicts
        messages = [{"role": r['role'], "content": r['content']} for r in rows]

        # Reverse to maintain chronological order (oldest first)
        messages.reverse()

        # Cache in Redis for 60 minutes
        self.redis.setex(cache_key, 3600, json.dumps({"messages": messages}))
        return {"messages": messages}
    
    def put_thread_message(self, thread_id, new_messages):
        cursor = self.conn.cursor()
        for msg in new_messages:
            cursor.execute("""
                INSERT INTO thread_messages (thread_id, role, content)
                VALUES (%s, %s, %s)
            """, (thread_id, msg["role"], msg["content"]))
        self.conn.commit()

        # Update Redis cache
        cache_key = f"thread:{thread_id}:thread_messages"
        self.redis.delete(cache_key)

    def create_new_thread(self, user_id, title):
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO thread (user_id, title) VALUES (%s, %s) RETURNING id
    """, (user_id, title))
        thread_id = cursor.fetchone()['id']
        self.conn.commit()

        # Update Redis cache
        cache_key = f"user:{user_id}:last_thread"
        self.redis.delete(cache_key)

        return str(thread_id)
    
    def get_last_thread(self, user_id):
        # Query DB
        cursor = self.conn.cursor()
        cursor.execute("""
                SELECT id FROM thread
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id,))
        row = cursor.fetchone()
        thread_id = row['id'] if row else None
        return thread_id