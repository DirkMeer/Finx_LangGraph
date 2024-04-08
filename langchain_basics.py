from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from setup_environment import set_environment_variables

set_environment_variables("Simple LangChain test")


french_german_prompt = ChatPromptTemplate.from_template(
    "Please tell me the french and german words for {word} with an example sentence for each."
)


llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
output_parser = StrOutputParser()

french_german_chain = french_german_prompt | llm | output_parser  # LCEL

# result = french_german_chain.invoke({"word": "polar bear"})
# print(result)

# for chunk in french_german_chain.stream({"word": "polar bear"}):
#     print(chunk, end="", flush=True)

# print(
#     french_german_chain.batch(
#         [{"word": "computer"}, {"word": "elephant"}, {"word": "carrot"}]
#     )
# )

# print("input_schema", french_german_chain.input_schema.schema())
# print("output_schema", french_german_chain.output_schema.schema())


check_if_correct_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that looks at a question and it's given answer. You will find out what is wrong with the answer and improve it. You will return the improved version of the answer.
    Question:\n{question}\nAnswer Given:\n{initial_answer}\nReview the answer give me an improved version instead.
    Improved answer:
    """
)

check_answer_chain = check_if_correct_prompt | llm | output_parser


def run_chain(word: str) -> str:
    initial_answer = french_german_chain.invoke({"word": word})
    print("initial answer:", initial_answer, end="\n\n")
    answer = check_answer_chain.invoke(
        {
            "question": f"Please tell me the french and german words for {word} with an example sentence for each.",
            "initial_answer": initial_answer,
        }
    )
    print("improved answer:", answer)
    return answer


run_chain("strawberries")
