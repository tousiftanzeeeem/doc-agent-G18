# index is built inside build_knowledge_base; kept for staged runs
from doc_agent import config, pipeline
pipeline.build_knowledge_base(config.load())
