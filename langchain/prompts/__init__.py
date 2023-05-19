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
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.few_shot_with_templates import FewShotPromptWithTemplates
from langchain.prompts.loading import load_prompt
from langchain.prompts.prompt import Prompt, PromptTemplate
from langchain.prompts.database.cassandra import createCassandraPromptTemplate
from langchain.prompts.dependencyful_prompt import DependencyfulPromptTemplate

__all__ = [
    "BasePromptTemplate",
    "StringPromptTemplate",
    "load_prompt",
    "PromptTemplate",
    "FewShotPromptTemplate",
    "Prompt",
    "FewShotPromptWithTemplates",
    "ChatPromptTemplate",
    "MessagesPlaceholder",
    "HumanMessagePromptTemplate",
    "AIMessagePromptTemplate",
    "SystemMessagePromptTemplate",
    "ChatMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "DependencyfulPromptTemplate",
    "createCassandraPromptTemplate",
]
