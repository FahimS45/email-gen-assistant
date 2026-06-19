"""
graph.py

Assembles the LangGraph StateGraph for the email generation pipeline.
"""

from langgraph.graph import StateGraph, END

from backend.graph.state import EmailState
from backend.graph.nodes import validate_input, build_prompt, generate_email


def _has_error(state: EmailState) -> str:
    """Conditional edge: routes to END early if validation failed."""
    return "end" if state.get("error") else "continue"


def build_graph() -> StateGraph:
    """
    Builds and compiles the email generation graph.
    Returns a compiled LangGraph app ready for .ainvoke() or .astream().
    """
    graph = StateGraph(EmailState)

    # Register nodes
    graph.add_node("validate_input",  validate_input)
    graph.add_node("build_prompt",    build_prompt)
    graph.add_node("generate_email",  generate_email)

    # Entry point
    graph.set_entry_point("validate_input")

    # Conditional edge after validation
    graph.add_conditional_edges(
        "validate_input",
        _has_error,
        {
            "end":      END,
            "continue": "build_prompt",
        },
    )

    # Linear edges for the rest
    graph.add_edge("build_prompt",   "generate_email")
    graph.add_edge("generate_email", END)

    return graph.compile()


# Singleton instance of the graph, to be imported and used by main.py and run_cli.py
email_graph = build_graph()
