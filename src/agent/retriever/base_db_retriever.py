import sqlite3
from abc import abstractmethod, ABC
from typing import List, Dict, Any, Tuple

class BaseDBRetriever(ABC):
    @abstractmethod
    def get_db_path(self) -> str:
        pass

    def _query(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.get_db_path())
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()