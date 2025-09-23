from typing import Any, Dict, Optional
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from langchain_core.messages import messages_from_dict, message_to_dict
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")


class MongoDBSaver(BaseCheckpointSaver):
    """MongoDB-based checkpoint saver for LangGraph."""

    def __init__(self, uri: str = MONGODB_URI, db_name: str = "langgraph", collection: str = "checkpoints"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.col = self.db[collection]

    # ------------------------
    # Internal serialization helpers
    # ------------------------
    def _serialize_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Convert messages into serializable dicts."""
        cp = checkpoint.copy()
        if "channel_values" in cp:
            for key, value in cp["channel_values"].items():
                if isinstance(value, dict) and "messages" in value:
                    cp["channel_values"][key]["messages"] = [
                        message_to_dict(m) for m in value["messages"]
                    ]
        return cp

    def _deserialize_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dicts back into LangChain messages."""
        cp = checkpoint.copy()
        if "channel_values" in cp:
            for key, value in cp["channel_values"].items():
                if isinstance(value, dict) and "messages" in value:
                    cp["channel_values"][key]["messages"] = messages_from_dict(
                        value["messages"]
                    )
        return cp

    # ------------------------
    # LangGraph required methods
    # ------------------------
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save or update a checkpoint in MongoDB."""
        thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id", "default-checkpoint")

        doc = {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint": self._serialize_checkpoint(checkpoint),
            "metadata": metadata or {},
        }

        self.col.update_one(
            {"thread_id": thread_id, "checkpoint_id": checkpoint_id},
            {"$set": doc},
            upsert=True,
        )

    def get(self, config: Dict[str, Any]) -> Optional[Checkpoint]:
        """Retrieve a checkpoint from MongoDB."""
        thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id", "default-checkpoint")

        doc = self.col.find_one({"thread_id": thread_id, "checkpoint_id": checkpoint_id})
        if not doc:
            return None
        return self._deserialize_checkpoint(doc["checkpoint"])

    def list(self, config: Optional[Dict[str, Any]] = None):
        """List checkpoints (for debugging or history browsing)."""
        query = {}
        if config:
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id:
                query["thread_id"] = thread_id
        for doc in self.col.find(query):
            yield self._deserialize_checkpoint(doc["checkpoint"])
