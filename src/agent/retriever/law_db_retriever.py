from typing import List, Dict, Any, Tuple

from src.agent.retriever.base_db_retriever import BaseDBRetriever
from src.config import config

class LawDBRetriever(BaseDBRetriever):
    def get_db_path(self) -> str:
        return config.LAW_DB_PATH

    def get_all_laws(self) -> List[str]:
        sql = """
        SELECT DISTINCT law_name FROM articles WHERE status IN ('现行有效','已被修改')
        """
        rows = super()._query(sql)
        return [r["law_name"] for r in rows]

    def get_articles(self, article_queries: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        if not article_queries:
            return []
        conditions = " OR ".join(["(law_name = ? AND article_no = ?)"] * len(article_queries))
        params = []
        for law_id, art_no in article_queries:
            params.append(law_id)
            params.append(art_no)

        sql = f"""
        SELECT law_id, law_name, chapter, section, article_no, article_text, article_no_arabic
        FROM articles
        WHERE {conditions}
        """
        rows = super()._query(sql, tuple(params))
        mapping = {(r["law_name"], r["article_no"]): r for r in rows}
        return [mapping[item] for item in article_queries if item in mapping]
