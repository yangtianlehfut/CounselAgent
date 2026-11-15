from typing import Dict, List

from src.config import config
from src.agent.retriever.base_db_retriever import BaseDBRetriever


class CrimeNotFoundError(LookupError):
    def __init__(self, crime_name: str):
        self.crime_name = crime_name
        super().__init__(f"罪名不存在：{crime_name}")


class CrimeDBRetriever(BaseDBRetriever):
    def get_db_path(self) -> str:
        return config.CRIME_DB_PATH

    def get_all_crime(self) -> List[Dict[str, str]]:
        sql = """
        SELECT name, article, constitution, non_constitution
        FROM crime
        ORDER BY id
        """
        return super()._query(sql)

    def get_core_factors(self, crime_name: str) -> List[Dict[str, str]]:
        sql = """
        SELECT q.keyword, q.question
        FROM crime AS c
        LEFT JOIN crime_question AS q ON q.crime_id = c.id
        WHERE c.name = ?
        ORDER BY q.id
        """
        rows = super()._query(sql, (crime_name,))
        if not rows:
            raise CrimeNotFoundError(crime_name)

        return [
            {"keyword": row["keyword"], "question": row["question"]}
            for row in rows
            if row["keyword"] is not None
        ]

