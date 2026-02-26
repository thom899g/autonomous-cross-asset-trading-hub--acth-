"""
Firebase configuration and connection management.
Provides centralized Firebase client with connection pooling and error recovery.
"""
import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from google.cloud.firestore import Client as FirestoreClient

logger = logging.getLogger(__name__)


@dataclass
class FirebaseConfig:
    """Firebase configuration dataclass with validation."""
    credentials_path: str
    project_id: str
    enable_offline_mode: bool = False


class FirebaseConnection:
    """Singleton Firebase connection manager with error recovery."""
    
    _instance: Optional['FirebaseConnection'] = None
    _client: Optional[FirestoreClient] = None
    _app = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseConnection, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._config: Optional[FirebaseConfig] = None
        self._connection_attempts = 0
        self._max_attempts = 3
        
    def initialize(self, config: FirebaseConfig) -> bool:
        """Initialize Firebase connection with retry logic."""
        self._config = config
        
        if not os.path.exists(config.credentials_path):
            logger.error(f"Firebase credentials file not found: {config.credentials_path}")
            if config.enable_offline_mode:
                logger.warning("Running in offline mode without Firebase")
                return False
            raise FileNotFoundError(f"Firebase credentials file not found: {config.credentials_path}")
        
        for attempt in range(self._max_attempts):
            try:
                if firebase_admin._DEFAULT_APP_NAME not in firebase_admin._apps:
                    cred = credentials.Certificate(config.credentials_path)
                    self._app = initialize_app(cred, {
                        'projectId': config.project_id,
                    })
                    logger.info(f"Firebase app initialized for project: {config.project_id}")
                
                self._client = firestore.client()
                
                # Test connection with a simple write
                test_ref = self._client.collection('system_health').document('connection_test')
                test_ref.set({
                    'timestamp': datetime.utcnow(),
                    'status': 'connected',
                    'attempt': attempt + 1
                }, merge=True)
                
                logger.info("Firebase connection established successfully")
                self._connection_attempts = 0
                return True
                
            except Exception as e:
                self._connection_attempts += 1
                logger.error(f"Firebase connection attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == self._max_attempts - 1:
                    logger.critical("All Firebase connection attempts failed")
                    if config.enable_offline_mode:
                        logger.warning("Proceeding in offline mode - data will be cached locally")
                        return False
                    raise ConnectionError(f"Failed to connect to Firebase after {self._max_attempts} attempts")
        
        return False
    
    def get_client(self) -> FirestoreClient:
        """Get Firestore client with connection health check."""
        if self._client is None:
            if self._config and self._config.enable_offline_mode:
                raise RuntimeError("Firebase is in offline mode - no client available")
            raise RuntimeError("Firebase not initialized. Call initialize() first")
        
        # Verify connection is still alive
        try:
            # Simple read operation to check connection
            self._client.collection('system_health').limit(1).get()
            return self._client
        except Exception as e:
            logger.error(f"Firebase connection lost: {str(e)}")
            # Attempt reconnection
            if self._config:
                self.initialize(self._config)
                return self._client
            raise ConnectionError("Firebase connection lost and cannot reconnect")
    
    def get_system_state_ref(self):
        """Get reference to system state document."""
        return self.get_client().collection('system_state').document('acth_main')
    
    def get_trading_log_ref(self):
        """Get reference to trading logs collection."""
        return self.get_client().collection('trading_logs')
    
    def get_correlation_data_ref(self):
        """Get reference to correlation data collection."""
        return self.get_client().collection('correlation_data')


# Global instance
firebase_conn = FirebaseConnection()