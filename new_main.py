import faiss
import numpy as np
import sqlite3
import pickle
import os
import json
import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Tuple, Optional, Any

class VectorDB:
    def __init__(self, db_path: str, table_name: str, agent_id: str, dimension: int = 384):
        """Initialize the vector database with SQL storage"""
        self.db_path = db_path
        self.table_name = table_name
        self.agent_id = agent_id
        self.dimension = dimension
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Default encoder
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        
        # Initialize the SQL database
        self._init_sql_db()
        
        # Load existing index if available
        self._load_index_from_sql()
    
    def _init_sql_db(self):
        """Initialize the SQL database and create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for vector indexes
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INTEGER PRIMARY KEY,
            agent_id TEXT,
            vector BLOB,
            metadata TEXT
        )
        ''')
        
        # Create table for faiss index blob
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS faiss_indexes (
            agent_id TEXT PRIMARY KEY,
            index_data BLOB
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_index_from_sql(self):
        """Load the FAISS index from SQL if it exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT index_data FROM faiss_indexes WHERE agent_id = ?", (self.agent_id,))
        result = cursor.fetchone()
        
        if result:
            index_bytes = result[0]
            self.index = faiss.deserialize_index(pickle.loads(index_bytes))
            
            # Also load all vectors
            cursor.execute(f"SELECT id, vector, metadata FROM {self.table_name} WHERE agent_id = ?", 
                          (self.agent_id,))
            self.data = {}
            for row in cursor.fetchall():
                doc_id, vector_bytes, metadata_str = row
                self.data[doc_id] = {
                    'vector': pickle.loads(vector_bytes),
                    'metadata': pickle.loads(metadata_str)
                }
        else:
            self.data = {}
        
        conn.close()
    
    def _save_index_to_sql(self):
        """Save the FAISS index to SQL"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize the index
        index_bytes = pickle.dumps(faiss.serialize_index(self.index))
        
        # Store or update the index
        cursor.execute(f'''
        INSERT OR REPLACE INTO faiss_indexes (agent_id, index_data)
        VALUES (?, ?)
        ''', (self.agent_id, index_bytes))
        
        conn.commit()
        conn.close()
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the vector database"""
        if metadata is None:
            metadata = [{}] * len(documents)
        
        # Encode documents
        vectors = self.encoder.encode(documents)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add vectors to FAISS index
        self.index.add(vectors.astype('float32'))
        
        # Store vectors and metadata in SQL
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            doc_id = len(self.data) + i
            vector_bytes = pickle.dumps(vector)
            meta_bytes = pickle.dumps(meta)
            
            cursor.execute(f'''
            INSERT INTO {self.table_name} (id, agent_id, vector, metadata)
            VALUES (?, ?, ?, ?)
            ''', (doc_id, self.agent_id, vector_bytes, meta_bytes))
            
            self.data[doc_id] = {
                'vector': vector,
                'metadata': meta
            }
        
        conn.commit()
        conn.close()
        
        # Save the updated index
        self._save_index_to_sql()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        query_vector = self.encoder.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search in the FAISS index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Retrieve results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.data):
                continue
                
            results.append({
                'id': int(idx),
                'distance': float(distance),
                'metadata': self.data[idx]['metadata']
            })
        
        return results


class UserHistoryManager:
    def __init__(self, db_path: str):
        """Initialize the user history manager with a path to the SQL database"""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the user history tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for user sessions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
        ''')
        
        # Create table for user queries
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_queries (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id TEXT,
            query_text TEXT,
            agent_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
        )
        ''')
        
        # Create table for query results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER,
            agent_id TEXT,
            doc_id INTEGER,
            distance REAL,
            metadata TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES user_queries(query_id)
        )
        ''')
        
        # Create table for user feedback
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER,
            result_id INTEGER,
            rating INTEGER,
            feedback_text TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES user_queries(query_id),
            FOREIGN KEY (result_id) REFERENCES query_results(result_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, user_id: str, metadata: Dict = None) -> str:
        """Create a new user session and return the session ID"""
        session_id = f"session_{user_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        metadata_str = json.dumps(metadata) if metadata else "{}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO user_sessions (session_id, user_id, metadata)
        VALUES (?, ?, ?)
        ''', (session_id, user_id, metadata_str))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def update_session(self, session_id: str, metadata: Dict = None) -> bool:
        """Update an existing user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if session exists
        cursor.execute("SELECT session_id FROM user_sessions WHERE session_id = ?", (session_id,))
        if cursor.fetchone() is None:
            conn.close()
            return False
        
        # Update session
        updates = ["last_active_at = CURRENT_TIMESTAMP"]
        params = []
        
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
        
        cursor.execute(f'''
        UPDATE user_sessions 
        SET {", ".join(updates)}
        WHERE session_id = ?
        ''', params + [session_id])
        
        conn.commit()
        conn.close()
        return True
    
    def log_query(self, session_id: str, user_id: str, query_text: str, agent_id: str) -> int:
        """Log a user query and return the query ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO user_queries (session_id, user_id, query_text, agent_id)
        VALUES (?, ?, ?, ?)
        ''', (session_id, user_id, query_text, agent_id))
        
        query_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return query_id
    
    def log_results(self, query_id: int, agent_id: str, results: List[Dict]) -> List[int]:
        """Log query results and return the result IDs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        result_ids = []
        for result in results:
            metadata_str = json.dumps(result.get('metadata', {}))
            
            cursor.execute('''
            INSERT INTO query_results (query_id, agent_id, doc_id, distance, metadata)
            VALUES (?, ?, ?, ?, ?)
            ''', (query_id, agent_id, result['id'], result['distance'], metadata_str))
            
            result_ids.append(cursor.lastrowid)
        
        conn.commit()
        conn.close()
        
        return result_ids
    
    def add_feedback(self, query_id: int, result_id: int, rating: int, feedback_text: str = "") -> int:
        """Add user feedback for a query result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO user_feedback (query_id, result_id, rating, feedback_text)
        VALUES (?, ?, ?, ?)
        ''', (query_id, result_id, rating, feedback_text))
        
        feedback_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return feedback_id
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent user query history"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT 
            q.query_id, q.query_text, q.agent_id, q.timestamp,
            COUNT(r.result_id) as result_count,
            AVG(f.rating) as avg_rating
        FROM user_queries q
        LEFT JOIN query_results r ON q.query_id = r.query_id
        LEFT JOIN user_feedback f ON r.result_id = f.result_id
        WHERE q.user_id = ?
        GROUP BY q.query_id
        ORDER BY q.timestamp DESC
        LIMIT ?
        ''', (user_id, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'query_id': row['query_id'],
                'query_text': row['query_text'],
                'agent_id': row['agent_id'],
                'timestamp': row['timestamp'],
                'result_count': row['result_count'],
                'avg_rating': row['avg_rating']
            })
        
        conn.close()
        return results
    
    def get_query_details(self, query_id: int) -> Dict:
        """Get detailed information about a specific query"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get query info
        cursor.execute('''
        SELECT * FROM user_queries WHERE query_id = ?
        ''', (query_id,))
        query_row = cursor.fetchone()
        
        if query_row is None:
            conn.close()
            return None
        
        query_info = {
            'query_id': query_row['query_id'],
            'session_id': query_row['session_id'],
            'user_id': query_row['user_id'],
            'query_text': query_row['query_text'],
            'agent_id': query_row['agent_id'],
            'timestamp': query_row['timestamp'],
            'results': []
        }
        
        # Get results info
        cursor.execute('''
        SELECT r.*, f.rating, f.feedback_text
        FROM query_results r
        LEFT JOIN user_feedback f ON r.result_id = f.result_id
        WHERE r.query_id = ?
        ORDER BY r.distance
        ''', (query_id,))
        
        for row in cursor.fetchall():
            result = {
                'result_id': row['result_id'],
                'doc_id': row['doc_id'],
                'agent_id': row['agent_id'],
                'distance': row['distance'],
                'metadata': json.loads(row['metadata']),
                'timestamp': row['timestamp'],
                'feedback': {
                    'rating': row['rating'],
                    'text': row['feedback_text']
                } if row['rating'] is not None else None
            }
            query_info['results'].append(result)
        
        conn.close()
        return query_info


class VectorDBManager:
    def __init__(self, db_path: str, table_name: str = "documents"):
        """Initialize the Vector DB Manager with a path to the SQL database"""
        self.db_path = db_path
        self.table_name = table_name
        self.vector_dbs = {}
        self.active_db = None
        
        # Initialize user history manager
        self.history_manager = UserHistoryManager(db_path)
        
        # Initialize the database connection and load available agent IDs
        self._init_db()
        self._load_available_agents()
        
        # Current session information
        self.current_session = None
        self.current_user_id = None
    
    def _init_db(self):
        """Initialize the SQL database if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for agent configuration
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vector_db_config (
            agent_id TEXT PRIMARY KEY,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_available_agents(self):
        """Load all available agent IDs from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT agent_id FROM faiss_indexes")
        agent_ids = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        # Initialize vector DBs for each agent
        for agent_id in agent_ids:
            self.vector_dbs[agent_id] = VectorDB(self.db_path, self.table_name, agent_id)
    
    def start_user_session(self, user_id: str, metadata: Dict = None) -> str:
        """Start a new user session"""
        session_id = self.history_manager.create_session(user_id, metadata)
        self.current_session = session_id
        self.current_user_id = user_id
        return session_id
    
    def update_session_metadata(self, metadata: Dict) -> bool:
        """Update the metadata for the current session"""
        if not self.current_session:
            return False
        return self.history_manager.update_session(self.current_session, metadata)
    
    def create_vector_db(self, agent_id: str, description: str = ""):
        """Create a new vector database with the specified agent ID"""
        if agent_id in self.vector_dbs:
            raise ValueError(f"Vector DB with agent_id '{agent_id}' already exists")
        
        # Create the vector DB
        vector_db = VectorDB(self.db_path, self.table_name, agent_id)
        self.vector_dbs[agent_id] = vector_db
        
        # Store agent info in config
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO vector_db_config (agent_id, description)
        VALUES (?, ?)
        ''', (agent_id, description))
        
        conn.commit()
        conn.close()
        
        # Set as active if it's the first DB
        if self.active_db is None:
            self.active_db = agent_id
        
        return vector_db
    
    def switch_db(self, agent_id: str) -> bool:
        """Switch to using a different vector database"""
        if agent_id not in self.vector_dbs:
            return False
        
        self.active_db = agent_id
        return True
    
    def get_active_db(self) -> Optional[VectorDB]:
        """Get the currently active vector database"""
        if self.active_db is None:
            return None
        return self.vector_dbs.get(self.active_db)
    
    def get_db(self, agent_id: str) -> Optional[VectorDB]:
        """Get a specific vector database by agent ID"""
        return self.vector_dbs.get(agent_id)
    
    def list_available_dbs(self) -> List[Dict]:
        """List all available vector databases"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT agent_id, description, created_at 
        FROM vector_db_config
        ORDER BY created_at
        ''')
        
        results = []
        for row in cursor.fetchall():
            agent_id, description, created_at = row
            results.append({
                'agent_id': agent_id,
                'description': description,
                'created_at': created_at,
                'is_active': agent_id == self.active_db
            })
        
        conn.close()
        return results
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None, agent_id: str = None):
        """Add documents to a vector database"""
        target_db = None
        
        # Determine target database
        if agent_id is not None:
            target_db = self.vector_dbs.get(agent_id)
            if target_db is None:
                raise ValueError(f"No vector DB found with agent_id '{agent_id}'")
        elif self.active_db is not None:
            target_db = self.vector_dbs[self.active_db]
        else:
            raise ValueError("No active vector DB and no agent_id specified")
        
        # Add documents to the target database
        target_db.add_documents(documents, metadata)
    
    def search(self, query: str, top_k: int = 5, agent_id: str = None) -> Dict:
        """Search in vector database(s) and log the query"""
        results = {}
        target_agent_id = agent_id if agent_id is not None else self.active_db
        
        # Log the query if we have an active session
        query_id = None
        if self.current_session and self.current_user_id:
            query_id = self.history_manager.log_query(
                self.current_session, 
                self.current_user_id,
                query,
                target_agent_id or "all"
            )
        
        # Search in specific DB
        if agent_id is not None:
            if agent_id not in self.vector_dbs:
                raise ValueError(f"No vector DB found with agent_id '{agent_id}'")
            db_results = self.vector_dbs[agent_id].search(query, top_k)
            results[agent_id] = db_results
            
            # Log results
            if query_id:
                self.history_manager.log_results(query_id, agent_id, db_results)
        
        # Search in active DB
        elif self.active_db is not None:
            db_results = self.vector_dbs[self.active_db].search(query, top_k)
            results[self.active_db] = db_results
            
            # Log results
            if query_id:
                self.history_manager.log_results(query_id, self.active_db, db_results)
        
        # Search in all DBs
        else:
            for db_id, db in self.vector_dbs.items():
                db_results = db.search(query, top_k)
                results[db_id] = db_results
                
                # Log results
                if query_id:
                    self.history_manager.log_results(query_id, db_id, db_results)
        
        return results
    
    def add_feedback(self, query_id: int, result_id: int, rating: int, feedback_text: str = "") -> int:
        """Add user feedback for a search result"""
        return self.history_manager.add_feedback(query_id, result_id, rating, feedback_text)
    
    def get_user_history(self, user_id: str = None, limit: int = 10) -> List[Dict]:
        """Get user search history"""
        if user_id is None:
            if self.current_user_id is None:
                raise ValueError("No user ID specified and no active user")
            user_id = self.current_user_id
        
        return self.history_manager.get_user_history(user_id, limit)
    
    def get_query_details(self, query_id: int) -> Dict:
        """Get detailed information about a specific query"""
        return self.history_manager.get_query_details(query_id)
    
    def delete_db(self, agent_id: str) -> bool:
        """Delete a vector database"""
        if agent_id not in self.vector_dbs:
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete from config
        cursor.execute("DELETE FROM vector_db_config WHERE agent_id = ?", (agent_id,))
        
        # Delete from faiss_indexes
        cursor.execute("DELETE FROM faiss_indexes WHERE agent_id = ?", (agent_id,))
        
        # Delete documents
        cursor.execute(f"DELETE FROM {self.table_name} WHERE agent_id = ?", (agent_id,))
        
        conn.commit()
        conn.close()
        
        # Remove from memory
        del self.vector_dbs[agent_id]
        
        # Reset active DB if necessary
        if self.active_db == agent_id:
            if self.vector_dbs:
                self.active_db = next(iter(self.vector_dbs))
            else:
                self.active_db = None
        
        return True


# Example usage
def main():
    # Initialize the VectorDBManager
    db_path = "vector_databases.db"
    manager = VectorDBManager(db_path)
    
    # Start a user session
    user_id = "user123"
    session_id = manager.start_user_session(user_id, {"source": "example script"})
    print(f"Started session {session_id} for user {user_id}")
    
    # Create vector databases if they don't exist
    if "agent1" not in manager.vector_dbs:
        manager.create_vector_db("agent1", "General knowledge database")
    
    if "agent2" not in manager.vector_dbs:
        manager.create_vector_db("agent2", "Technical documentation database")
    
    # Example documents for agent1
    agent1_docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Vector databases help in semantic search applications.",
        "FAISS is a library for efficient similarity search."
    ]
    
    agent1_metadata = [
        {"source": "English proverb", "category": "language"},
        {"source": "AI textbook", "category": "technology"},
        {"source": "Database article", "category": "technology"},
        {"source": "Library documentation", "category": "technology"}
    ]
    
    # Add documents to agent1
    manager.switch_db("agent1")
    manager.add_documents(agent1_docs, agent1_metadata)
    
    # Example documents for agent2
    agent2_docs = [
        "Python is a popular programming language.",
        "SQLite is a lightweight database engine.",
        "Natural language processing deals with text analysis.",
        "Deep learning uses neural networks with multiple layers."
    ]
    
    agent2_metadata = [
        {"source": "Programming guide", "category": "technology"},
        {"source": "Database documentation", "category": "technology"},
        {"source": "NLP article", "category": "technology"},
        {"source": "AI research paper", "category": "technology"}
    ]
    
    # Switch to agent2 and add documents
    manager.switch_db("agent2")
    manager.add_documents(agent2_docs, agent2_metadata)
    
    # Perform searches with history tracking
    queries = [
        "artificial intelligence",
        "database systems",
        "programming languages",
        "natural language processing"
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = manager.search(query, top_k=2)
        
        for agent_id, agent_results in results.items():
            print(f"\nResults from {agent_id}:")
            for item in agent_results:
                print(f"ID: {item['id']}, Distance: {item['distance']:.4f}, Metadata: {item['metadata']}")
                
                # Add some random feedback
                if "AI" in query or "database" in query:
                    # Find the query_id from history
                    history = manager.get_user_history(user_id, limit=1)
                    if history:
                        query_id = history[0]['query_id']
                        result_id = manager.history_manager.get_query_details(query_id)['results'][0]['result_id']
                        rating = 5 if "AI" in query else 4
                        manager.add_feedback(query_id, result_id, rating, "Helpful result")
    
    # Display user history
    print("\nUser query history:")
    history = manager.get_user_history(user_id)
    for item in history:
        print(f"Query: '{item['query_text']}', Agent: {item['agent_id']}, Results: {item['result_count']}, Rating: {item['avg_rating']}")
    
    # Display detailed info for a specific query
    if history:
        query_id = history[0]['query_id']
        print(f"\nDetailed info for query_id {query_id}:")
        details = manager.get_query_details(query_id)
        print(f"Query: '{details['query_text']}'")
        print(f"User: {details['user_id']}")
        print(f"Session: {details['session_id']}")
        print(f"Agent: {details['agent_id']}")
        print("Results:")
        for result in details['results']:
            print(f"  Doc ID: {result['doc_id']}, Distance: {result['distance']:.4f}")
            if result['feedback']:
                print(f"  Feedback: {result['feedback']['rating']}/5 - '{result['feedback']['text']}'")


if __name__ == "__main__":
    main()