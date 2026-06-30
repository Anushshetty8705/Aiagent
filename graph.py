from langgraph.graph import StateGraph
from typing import TypedDict, Any, List, Dict

from agents.planner_agent import PlannerAgent
from tools.semantic_search import SemanticSearchTool
from agents.metadata_filter_agent import MetadataFilterAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.explanation_agent import ExplanationAgent


# =========================
# MEMORY STORE (simple in-memory DB)
# =========================
memory_store: List[Dict] = []


# =========================
# STATE
# =========================
class State(TypedDict):
    query: str
    plan: dict

    results: Any
    filtered: Any
    ranked: Any

    explanation: str

    rewritten_query: str
    is_good: bool
    iteration: int
    confidence: float

    memory: list


# =========================
# AGENTS
# =========================
planner = PlannerAgent()
search = SemanticSearchTool()
filter_agent = MetadataFilterAgent()
evaluator = EvaluatorAgent()
explainer = ExplanationAgent()


# =========================
# NODES
# =========================

def plan_node(state: State):
    state["plan"] = planner.plan(state["query"])
    state["iteration"] = 0
    return state


def search_node(state: State):
    query = state.get("rewritten_query") or state["query"]
    state["results"] = search.search(query)
    return state


def filter_node(state: State):
    state["filtered"] = filter_agent.filter(
        state["results"],
        state["plan"]
    )
    return state


def rank_node(state: State):
    state["ranked"] = evaluator.rank(
        state["filtered"],
        state["plan"]
    )
    return state


# =========================
# EVALUATION NODE
# =========================
def evaluate_node(state: State):
    df = state.get("ranked")

    if df is None or len(df) == 0:
        state["is_good"] = False
        state["confidence"] = 0.0
        return state

    top_score = df.iloc[0]["score"]

    state["confidence"] = float(top_score)
    state["is_good"] = top_score > 0.55

    return state


# =========================
# REFINER NODE (AUTONOMY)
# =========================
def refine_node(state: State):
    plan = state["plan"]

    state["rewritten_query"] = " ".join([
        state["query"],
        str(plan.get("brand", "")),
        str(plan.get("display", "")),
        str(plan.get("camera", "")),
        str(plan.get("ram", ""))
    ])

    state["iteration"] = state.get("iteration", 0) + 1
    return state


# =========================
# EXPLANATION NODE
# =========================
def explain_node(state: State):
    state["explanation"] = explainer.explain(
        state["plan"],
        state["ranked"]
    )
    return state


# =========================
# MEMORY NODE
# =========================
def memory_node(state: State):
    df = state.get("ranked")

    top = df.iloc[0].to_dict() if df is not None and len(df) > 0 else None

    memory_store.append({
        "query": state["query"],
        "plan": state["plan"],
        "confidence": state.get("confidence", 0.0),
        "top_result": top
    })

    state["memory"] = memory_store
    return state


# =========================
# ROUTER (AUTONOMY BRAIN)
# =========================
def route_after_eval(state: State):
    if state["is_good"]:
        return "explain"

    if state.get("iteration", 0) >= 2:
        return "explain"

    return "refine"


# =========================
# GRAPH BUILD
# =========================
graph = StateGraph(State)

# nodes
graph.add_node("planner", plan_node)
graph.add_node("search", search_node)
graph.add_node("filter", filter_node)
graph.add_node("rank", rank_node)
graph.add_node("evaluate", evaluate_node)
graph.add_node("refine", refine_node)
graph.add_node("explain", explain_node)
graph.add_node("memory", memory_node)

# entry
graph.set_entry_point("planner")

# main pipeline
graph.add_edge("planner", "search")
graph.add_edge("search", "filter")
graph.add_edge("filter", "rank")
graph.add_edge("rank", "evaluate")

# AUTONOMY LOOP (IMPORTANT FIX)
graph.add_conditional_edges(
    "evaluate",
    route_after_eval,
    {
        "refine": "refine",
        "explain": "explain"
    }
)

graph.add_edge("refine", "search")

# final step
graph.add_edge("explain", "memory")
graph.set_finish_point("memory")

app = graph.compile()