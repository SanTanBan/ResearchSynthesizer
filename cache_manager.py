from typing import Dict, Any, Optional
import time
import json

class CacheManager:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
    
    def _cleanup(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.cache.items()
            if current_time - value['timestamp'] > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache if it exists and is not expired"""
        self._cleanup()
        
        if key in self.cache:
            cache_entry = self.cache[key]
            if time.time() - cache_entry['timestamp'] <= self.ttl:
                return cache_entry['data']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Dict[str, Any]):
        """Set value in cache"""
        self._cleanup()
        
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'data': value,
            'timestamp': time.time()
        }
