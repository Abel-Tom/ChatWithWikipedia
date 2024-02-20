from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

def langchain_agent(query,openai_api_key):
    llm = OpenAI(temperature=0.5,openai_api_key=openai_api_key)
    tools = load_tools(["wikipedia"], llm=llm)

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    result = agent.run(
        query
    )
    return result

if __name__ == "__main__":
    langchain_agent('estimated epa of lucid air pure','')
