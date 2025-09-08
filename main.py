import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Jen-Hsun "Jensen" Huang[a] (Chinese: 黃仁勳; pinyin: Huáng Rénxūn; Tâi-lô: N̂g Jîn-hun; born February 17, 1963) is a Taiwanese and American businessman, electrical engineer, and philanthropist who is the president, co-founder, and chief executive officer (CEO) of Nvidia, the world's largest semiconductor company.[3] In 2025, Forbes estimated his net worth at US$150 billion, making Huang the sixth-wealthiest individual in the world.[4]

    The son of Taiwanese American immigrants, Huang spent his childhood in Taiwan and Thailand before moving to the United States, where he was a student in Kentucky and Oregon. After earning his master's degree from Stanford University, Huang launched Nvidia in 1993 from a local Denny's restaurant at age 30 and has remained president and CEO since its founding. He led the company out of near-bankruptcy during the 1990s and oversaw its expansion into GPU production, high-performance computing, and artificial intelligence (AI).

    Under Huang, Nvidia experienced rapid growth during the AI boom, becoming the first company to reach a market capitalization of $4.0 trillion in July 2025.[5] In 2021 and 2024, Time magazine named Huang as one of the most influential people in the world. 
    """


    summary_template = """
    You are a world class summarizer. You will be given a piece of information {information} about a person. and I want you to create:
    1. A short summary about the person in 2-3 sentences.
    2. A list of 3 interesting facts about the person."""


    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    openaillm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    #ollamallm = ChatOllama(model="gemma3:270m", temperature=0)

    chain = summary_prompt_template | openaillm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
