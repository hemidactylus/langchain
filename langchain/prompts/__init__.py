"""Prompt template classes."""
from langchain.prompts.base import BasePromptTemplate, StringPromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    BaseChatPromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.prompts.database.cassandra import createCassandraPromptTemplate
from langchain.prompts.database.feast import createFeastPromptTemplate
from langchain.prompts.dependencyful_prompt import DependencyfulPromptTemplate
from langchain.prompts.example_selector import (
    LengthBasedExampleSelector,
    MaxMarginalRelevanceExampleSelector,
    NGramOverlapExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.few_shot_with_templates import FewShotPromptWithTemplates
from langchain.prompts.loading import load_prompt
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import Prompt, PromptTemplate

__all__ = [
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BasePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    "createCassandraPromptTemplate",
    "createFeastPromptTemplate",
    "DependencyfulPromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptWithTemplates",
    "HumanMessagePromptTemplate",
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "MessagesPlaceholder",
    "NGramOverlapExampleSelector",
    "PipelinePromptTemplate",
    "Prompt",
    "PromptTemplate",
    "SemanticSimilarityExampleSelector",
    "StringPromptTemplate",
    "SystemMessagePromptTemplate",
    "load_prompt",
]
