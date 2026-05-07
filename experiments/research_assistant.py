# IMPORTING WHAT WE NEED TO BUILD THE GRAPH AND CALL THE LLM
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

# SETTING UP THE LLM WITH TEMPERATURE 0 SO IT GIVES CONSISTENT ANSWERS
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# DEFINING THE STATE OBJECT THAT ALL NODES SHARE AND PASS BETWEEN EACH OTHER
class GraphState(TypedDict):
    topic: str
    # REDUCER KEEPS HISTORY OF ALL STATE CHANGES. APPENDS WITH ADD OPERATOR
    facts: Annotated[list, operator.add]
    critique: str
    iteration: int
    final_report: str

def researcher(state: GraphState) -> GraphState:
    topic = state["topic"]
    facts = state["facts"]
    # ASK THE LLM FOR FACTS THEN BREAK ITS RESPONSE INTO A LIST WHERE EACH LINE IS ITS OWN ITEM IN THE LIST
    response = llm.invoke(f"Existing facts so far: {facts}. Don't repeat. Find 5 facts about {topic}. Return one fact per line, no numbering").content
    splitFacts = response.split("\n")
    return {"facts": splitFacts}

def critic(state: GraphState) -> GraphState:
    topic = state["topic"]
    facts = state["facts"]
    critique = llm.invoke(f"Are these facts: {facts} sufficient to write a report about {topic}? Reply 'sufficient' or 'insufficient' and explain.").content
    return {"critique": critique, "iteration": state["iteration"] + 1}

def writer(state: GraphState) -> GraphState:
    topic = state["topic"]
    facts = state["facts"]
    final_report = llm.invoke(f"Write a brief report on {topic} given these accumalated facts: {facts}").content
    return {"final_report": final_report}


























# ASKING THE USER FOR A QUESTION AND PRINTING ALL THE RESULTS
if __name__ == "__main__":
    result = graph.invoke({"topic": input("What topic would you like to research?")})