from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage

def echo(state: MessagesState):
    last = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"echo: {last}")]}

g = StateGraph(MessagesState)
g.add_node("echo", echo)
g.add_edge(START, "echo")
g.add_edge("echo", END)
app = g.compile()

if __name__ == "__main__":
    out = app.invoke({"messages": [HumanMessage(content="hi")]})
    print(out["messages"][-1].content)
