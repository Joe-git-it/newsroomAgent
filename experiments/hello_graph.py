# IMPORTING WHAT WE NEED TO BUILD THE GRAPH AND CALL THE LLM
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

# SETTING UP THE LLM WITH TEMPERATURE 0 SO IT GIVES CONSISTENT ANSWERS
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# DEFINING THE STATE OBJECT THAT ALL NODES SHARE AND PASS BETWEEN EACH OTHER
class GraphState(TypedDict):
    question: str
    questionType: str
    thought: str
    answer: str
    critique: str

# NODE 1 SENDS THE QUESTION TO THE LLM AND SAVES THE REASONING AS THOUGHT
def contextBuilder(state: GraphState) -> GraphState:
    thought = llm.invoke(state["question"]).content
    return {"thought": thought}

# NODE 2 LOOKS AT THE THOUGHT AND DECIDES IF THE QUESTION IS LOGICAL OR EMOTIONAL
def classifier(state: GraphState) -> GraphState:
    thought = state["thought"]
    question = state["question"]
    questionType = llm.invoke(f"Given the reasoning in {thought} determine if the original question {question} was of a logical nature or an emotional nature. respond in one word. Respond either 'emotional' or 'logical'").content
    return {"questionType": questionType}

# THIS READS THE QUESTION TYPE AND RETURNS WHICH NODE TO GO TO NEXT
def router(state: GraphState) -> str:
    if "logical" in state["questionType"].lower():
        return "logicResponder"
    return "emotionResponder"

# NODE 3A ANSWERS THE QUESTION FROM A SCIENTIST PERSPECTIVE
def logicResponder(state: GraphState) -> GraphState:
    thought = state["thought"]
    question = state["question"]
    answer = llm.invoke(f"Given the reasoning in {thought} answer {question} as if you are a scientist in one sentence. first state you are a scientist").content
    return {"answer": answer}

# NODE 3B ANSWERS THE QUESTION FROM A THERAPIST PERSPECTIVE
def emotionResponder(state: GraphState) -> GraphState:
    thought = state["thought"]
    question = state["question"]
    answer = llm.invoke(f"Given the reasoning in {thought} answer {question} as if you are a therapist in one sentence. first state you are a therapist").content
    return {"answer": answer}

# NODE 4 GRADES THE ANSWER AND TELLS THE USER HOW SATISFIED THEY SHOULD BE
def critic(state: GraphState) -> GraphState:
    thought = state["thought"]
    question = state["question"]
    answer = state["answer"]
    critique = llm.invoke(f"Given the reasoning in {thought} and the original question in {question}, in one sentance give the answer {answer} a grade on how satisified the user should be").content
    return {"critique": critique}

# REGISTERING ALL THE NODES WITH THE GRAPH BUILDER
builder = StateGraph(GraphState)
builder.add_node("contextBuilder", contextBuilder)
builder.add_node("classifier", classifier)
builder.add_node("emotionResponder", emotionResponder)
builder.add_node("logicResponder", logicResponder)
builder.add_node("critic", critic)

# WIRING UP THE EDGES SO THE GRAPH KNOWS WHAT ORDER TO RUN THINGS
builder.set_entry_point("contextBuilder")
builder.add_edge("contextBuilder", "classifier")
builder.add_conditional_edges("classifier", router)
builder.add_edge("logicResponder", "critic")
builder.add_edge("emotionResponder", "critic")
builder.add_edge("critic", END)

# COMPILING THE GRAPH SO IT IS READY TO RUN
graph = builder.compile()

# ASKING THE USER FOR A QUESTION AND PRINTING ALL THE RESULTS
if __name__ == "__main__":
    result = graph.invoke({"question": input("What is your question?")})
    print("Thought :", result["thought"])
    print("Answer  :", result["answer"])
    print("Critique  :", result["critique"])
