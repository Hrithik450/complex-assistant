import json

class ThreadService():
    def __init__(self, connection, redis_client):
        self.conn = connection
        self.redis = redis_client

    def createNewThread(self, user_id, title):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO thread (user_id, title) VALUES (%s, %s) RETURNING id
        """, (user_id, title))
        
        thread_id = cursor.fetchone()['id']
        self.conn.commit()

        # Update Redis cache
        cache_key = f"threads:{user_id}"
        self.redis.delete(cache_key)

        return str(thread_id)

    def updateThreadMessages(self, thread_id, new_messages):
        """
        Insert one or many messages for a given thread.

        new_messages can be:
        • a dict: {"role": "...", "content": "..."}
        • a list of dicts: [{"role": "...", "content": "..."}, ...]
        """
        # Normalize to a list so we can iterate
        if isinstance(new_messages, dict):
            messages_to_insert = [new_messages]
        else:
            # Ensure we keep the order that was passed in
            messages_to_insert = list(new_messages)

        cursor = self.conn.cursor()
        for msg in messages_to_insert:
            cursor.execute("""
                INSERT INTO thread_messages (thread_id, role, content)
                VALUES (%s, %s, %s)
            """, (thread_id, msg["role"], msg["content"]),
            )

        self.conn.commit()

        # Invalidate Redis cache so next read hits DB
        cache_key1 = f"thread:{thread_id}:all_thread_messages"
        cache_key2 = f"thread:{thread_id}:recent_thread_messages"
        self.redis.delete(cache_key1)
        self.redis.delete(cache_key2)

    def renameThread(self, thread_id, new_title):
        """Renames a specific thread."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE thread SET title = %s WHERE id = %s
        """, (new_title, thread_id))
        self.conn.commit()

    # We now fetch the creation date to show in the sidebar.
    def getThreads(self, user_id):
        # cache_key = f"threads:{user_id}"
        # cached_value = self.redis.get(cache_key)
        # if cached_value:
        #     return json.loads(cached_value)

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, title, created_at 
            FROM thread
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """, (user_id,))
        rows = cursor.fetchall()
        threads = [{"id": str(r['id']), "title": r['title'], "created_at": r['created_at']} for r in rows]

        # Cache in Redis for 60 minutes
        # self.redis.setex(cache_key, 3600, json.dumps(threads))
        return threads

    def getLastThread(self, user_id):
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

    def getThreadMessages(self, thread_id):
        # Check Redis cache first
        cache_key = f"thread:{thread_id}:all_thread_messages"
        cached_value = self.redis.get(cache_key)
        if cached_value:
            return json.loads(cached_value)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT role, content 
            FROM thread_messages
            WHERE thread_id = %s 
            ORDER BY created_at ASC
        """, (thread_id,))
        rows = cursor.fetchall()

        # Convert rows into list of dicts
        messages = [{"role": r['role'], "content": r['content']} for r in rows]

        # Cache in Redis for 60 minutes
        self.redis.setex(cache_key, 3600, json.dumps({"messages": messages}))
        return {"messages": messages}

    def getRecentThreadMessages(self, thread_id):
        # Check Redis cache first
        cache_key = f"thread:{thread_id}:recent_thread_messages"
        cached_value = self.redis.get(cache_key)
        if cached_value:
            return json.loads(cached_value)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT role, content 
            FROM thread_messages
            WHERE thread_id = %s 
            ORDER BY created_at DESC
            LIMIT 4
        """, (thread_id,))
        rows = cursor.fetchall()

        # Convert rows into list of dicts
        messages = [{"role": r['role'], "content": r['content']} for r in rows]

        # Reverse to maintain chronological order (oldest first)
        messages.reverse()

        # Cache in Redis for 60 minutes
        self.redis.setex(cache_key, 3600, json.dumps({"messages": messages}))
        return {"messages": messages}

    def deleteThread(self, thread_id):
        """Deletes a specific thread and all its messages."""
        cursor = self.conn.cursor()
        # The ON DELETE CASCADE in your database schema will handle deleting messages
        cursor.execute("""
            DELETE FROM thread WHERE id = %s
        """, (thread_id,))
        self.conn.commit()

        # Invalidate any caches related to this thread
        cache_key1 = f"thread:{thread_id}:all_thread_messages"
        cache_key2 = f"thread:{thread_id}:recent_thread_messages"
        self.redis.delete(cache_key1)
        self.redis.delete(cache_key2)