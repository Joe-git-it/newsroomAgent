from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
# from langchain_anthropic import ChatAnthropic
#TO LATER BE REPLACED WITH SQL BACKEND
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint
from langchain_ollama import ChatOllama, OllamaEmbeddings

# SETTING UP THE LLM WITH TEMPERATURE 0 SO IT GIVES CONSISTENT ANSWERS
# llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
llm = ChatOllama(model="llama3.2:3b", temperature=0)

e = OllamaEmbeddings(model="nomic-embed-text")


# DEFINING THE STATE OBJECT THAT ALL NODES SHARE AND PASS BETWEEN EACH OTHER
class GraphState(TypedDict):
    topic: str
    # REDUCER KEEPS HISTORY OF ALL STATE CHANGES. APPENDS WITH ADD OPERATOR
    facts: Annotated[list, operator.add]
    critique: str
    iteration: Annotated[int, operator.add]
    final_report: str

def researcher(state: GraphState) -> GraphState:
    print("RESEARCHER RUNNING")
    topic = state["topic"]
    facts = state["facts"]
    # ASK THE LLM FOR FACTS THEN BREAK ITS RESPONSE INTO A LIST WHERE EACH LINE IS ITS OWN ITEM IN THE LIST
    response = llm.invoke(f"Existing facts so far: {facts}. Don't repeat. Find 5 facts about {topic}. Return one fact per line, no numbering").content
    print(response)
    splitFacts = response.split("\n")
    return {"facts": splitFacts}

def critic(state: GraphState) -> GraphState:
    print("CRITIC RUNNING")
    topic = state["topic"]
    facts = state["facts"]
    critique = llm.invoke(f"Looking at the facts gathered: {facts} reply in the single lower case word 'approve' if there are 5 distinct facts on {topic}. otherwise reply 'reject'").content
    print(critique)
    return {"critique": critique, "iteration": 1}

def writer(state: GraphState) -> GraphState:
    print("WRITER RUNNING")
    topic = state["topic"]
    facts = state["facts"]
    critique = state["critique"]
    final_report = llm.invoke(f"If {critique} is 'reject', explain that a report cannot be generated in one sentence. If {critique} is 'approve', write a brief report on {topic} given these accumalated facts: {facts}.").content
    return {"final_report": final_report}

#ROUTER
def should_continue(state) -> str:
    if state["iteration"] >= 3:
        return "writer"
    if "approve" in state["critique"].lower():
        return "writer"
    return "researcher"

# REGISTERING ALL THE NODES WITH THE GRAPH BUILDER
builder = StateGraph(GraphState)
builder.add_node("researcher", researcher)
builder.add_node("critic", critic)
builder.add_node("writer", writer)

# WIRING UP THE EDGES SO THE GRAPH KNOWS WHAT ORDER TO RUN THINGS
builder.set_entry_point("researcher")
builder.add_edge("researcher", "critic")
builder.add_conditional_edges("critic", should_continue)
builder.add_edge("writer", END)


# COMPILING THE GRAPH WITH AN IN MEMORY CHECKPOINTER SO STATE PERSISTS BETWEEN STEPS
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


if __name__ == "__main__":
     # THREAD ID IDENTIFIES THIS RUN. WILL REUSE IT TO CONTINUE THE SAME STATE THREAD
    config = {"configurable": {"thread_id": "session-1"}}
    result = graph.invoke({"topic": input("What topic would you like to research? ")}, config=config)
    print(result["iteration"])
    print(result["final_report"])

    # CURRENT STATE SNAPSHOT FOR THIS THREAD
    current = graph.get_state(config)
    print("CURRENT STATE VALUES:", current.values)
    print("NEXT NODE TO RUN:", current.next)
    #PRINT HISTORY, NEWEST FIRST
    for snapshot in graph.get_state_history(config):
        pprint(snapshot.values)