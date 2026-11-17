import keyword
import time

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agent import workflow
from src.debug.mock_nodes import get_mock_node_overrides
from src.logger import get_logger
from src.retriever.crime_db_retriever import CrimeDBRetriever
from src.retriever.law_db_retriever import LawDBRetriever


logger = get_logger(__name__)

# queries = [
#     ("中华人民共和国刑法", "第一条"),
#     ("中华人民共和国刑事诉讼法", "第二百九十八条"),
#     ("中华人民共和国全民所有制工业企业法", "第六十四条")
# ]
# result = LawDBRetriever().get_articles(queries)
# for article in result:
#     logger.info("Law article: %s", article)
#
# retriever = CrimeDBRetriever()
# all_crime = retriever.get_all_crime()
# logger.info("All crimes: %s", all_crime)
# keyword = retriever.get_core_factors('危险驾驶罪')
# logger.info("Core factors for dangerous driving: %s", keyword)

query1 = "帮我查一下刑法第100条是什么内容"
# query2 = '''
# 上个月有天晚上，我跟几个老同学在城南烧烤摊聚了一下，好久没见就喝了点酒，我开车去的，本来没打算喝，
# 架不住劝就喝了两瓶多啤酒。散场的时候觉得自己挺清醒的，家就隔两条街，想着这么晚了不会那么巧被查到，就开车走了。
# 结果拐上大路没开多远，前面就有交警设卡查酒驾，一排锥桶摆着挨个吹气。我当时想掉头后面有车顶着，靠边也没位子，
# 只能硬着头皮往前开。轮到我吹的时候我还想糊弄，说刚吃了槟榔，交警没理我，让我下车重新吹，仪器直接响了。
# 后来被拉去医院抽血，过了几天结果出来，血液酒精含量一百四十多。有没有帮我找一下类似案例。
# '''
# query3 = '''
# 律师您好，我老公昨晚被交警带走了，到现在还没回来，我该怎么办啊？
# 说是喝酒开车了，具体喝了多少我也不知道，他跟同事吃饭，我也没在场。反正就是开车回家路上好像蹭了别人的车，
# 对方报警了，交警来测了酒精，说超标了，人就被带走了。现在我也不知道他在哪个交警队，打电话也不接。
# 我听人说这个要判刑的，是真的吗？他就是个普通上班的，从来没犯过事，驾照也一直好好的。家里孩子还小，
# 他要真进去了我们娘俩怎么办啊。我现在能去看他吗？还是得等通知？要不要赶紧请律师？请律师有用吗？我什么都不懂，
# 您跟我说说我先干什么。
# '''

workflow = workflow.build_graph(checkpointer=MemorySaver(), node_overrides=get_mock_node_overrides())

timestamp = time.strftime("%m%d%H%M%S")
thread_config = {"configurable": {"thread_id": timestamp}}
logger.debug('start query1')
while True:
    user_query = input('请输入：')
    user_query = query1 if user_query == '' else user_query
    response = workflow.invoke({'messages': [HumanMessage(user_query)]}, config=thread_config)
    logger.info(response["final_response"])

# timestamp = time.strftime("%m%d%H%M%S")
# thread_config = {"configurable": {"thread_id": timestamp}}
# logger.debug('start query2')
# workflow.invoke({'messages': [HumanMessage(query2)]}, config=thread_config)
#
# timestamp = time.strftime("%m%d%H%M%S")
# thread_config = {"configurable": {"thread_id": timestamp}}
# logger.debug('start query3')
# workflow.invoke({'messages': [HumanMessage(query3)]}, config=thread_config)
