from langchain.prompts.database.converter_prompt_template import __all__

EXPECTED_ALL = [
    "ConverterType",
    "ConverterPromptTemplate",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
