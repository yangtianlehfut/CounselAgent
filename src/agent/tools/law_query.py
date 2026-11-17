"""LangChain tools backed by the law database retriever."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.retriever.law_db_retriever import LawDBRetriever


class ArticleQuery(BaseModel):
    """An exact law and article-number lookup."""

    law_name: str = Field(
        description=(
            "法律正式完整名称，不得使用简称、俗称或自行补改的版本名称，不需要带《》。例如使用"
            "“中华人民共和国刑法”而不是“刑法”，使用"
            "“中华人民共和国刑法修正案（十二）”而不是"
            "“中华人民共和国刑法修正案十二”。"
        )
    )
    article_no: str = Field(
        description=(
            "数据库中登记的原始中文条文编号，必须保留原格式，不要转换成"
            "阿拉伯数字。支持以下四种格式："
            "A 标准法条，例如“第一百零一条”；"
            "B 之 X 附加条文，例如“第一百二十条之二”；"
            "C 自定义条目系列，例如“条目五十一”；"
            "D 全文标记，使用“全文”。"
        )
    )


class GetArticlesInput(BaseModel):
    """Input schema for a batch of exact article lookups."""

    queries: list[ArticleQuery] = Field(
        min_length=1,
        description="需要查询的法律名称和条文编号，可一次传入多条。",
    )


_retriever = LawDBRetriever()


@tool("get_articles", args_schema=GetArticlesInput)
def get_articles(queries: list[ArticleQuery]) -> list[str]:
    """按准确的法律名称和中文条文编号批量查询法条原文及其章节信息。

    支持的法律名称覆盖：
    1. 宪法及宪法相关法、民法商法、行政法、经济法、社会法、刑法，以及
       诉讼与非诉讼程序法等主要法律部门中的中华人民共和国法律；
    2. 中华人民共和国宪法修正案和中华人民共和国刑法修正案；
    3. 全国人民代表大会常务委员会作出的法律解释；
    4. 全国人民代表大会常务委员会关于修改法律、批准、授权及法律问题的
       决定和决议，以及军事、特别行政区等相关法律文件。

    法律名称必须使用完整名称，例如“中华人民共和国刑法”“中华人民共和国
    民法典”“中华人民共和国刑法修正案（十二）”。数据库中每项记录的正式
    全称以 get_all_laws 的返回值为准；仅在不确定法律全称或查询没有匹配
    结果时才调用 get_all_laws。
    """
    article_queries = [
        (query.law_name, query.article_no)
        for query in queries
    ]
    return [row['article_text'] for row in _retriever.get_articles(article_queries)]


@tool("get_all_laws")
def get_all_laws() -> list[str]:
    """列出法条数据库当前支持查询的法律全称。"""
    return _retriever.get_all_laws()


LAW_QUERY_TOOLS = (get_articles, get_all_laws)
