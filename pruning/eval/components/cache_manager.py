import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.cache_file = self.cache_dir / ".eval_cache.pkl"

        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(self, model_info: Dict) -> str:
        if not self.enabled:
            return ""

        model_path = Path(model_info["path"])
        if model_path.exists():
            mtime = model_path.stat().st_mtime
            key_str = (
                f"{model_info['path']}_"
                f"{mtime}_"
                f"{model_info.get('strategy', '')}_"
                f"{model_info.get('pruning_ratio', '')}"
            )
            return hashlib.md5(key_str.encode()).hexdigest()
        return ""

    def load_cache(self) -> Dict:
        if not self.enabled or not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, "rb") as f:
                cache = pickle.load(f)
            logger.info(f"Loaded evaluation cache with {len(cache)} entries")
            return cache
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return {}

    def save_cache(self, cache: Dict) -> None:
        if not self.enabled:
            return

        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get_cached_result(
        self, model_info: Dict, cache: Dict
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        cache_key = self.generate_cache_key(model_info)
        if cache_key and cache_key in cache:
            logger.info(f"Using cached results for {model_info['dir_name']}")
            return cache[cache_key]
        return None

    def store_result(
        self, model_info: Dict, result: Dict[str, Any], cache: Dict
    ) -> None:
        if not self.enabled:
            return

        cache_key = self.generate_cache_key(model_info)
        if cache_key:
            cache[cache_key] = result
            self.save_cache(cache)
