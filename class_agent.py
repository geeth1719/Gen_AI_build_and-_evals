import json
from typing import List, Dict, Any

class AutonomousAgent:
    def __init__(self, agent_name: str, tools: Dict[str, callable]):
        self.name = agent_name
        self.tools = tools
        self.memory: List[Dict[str, str]] = []

    def plan_and_execute(self, objective: str, max_steps: int = 5):
        print(f"🎯 Objective: {objective}\n")
        
        # Initial prompt sets the ReAct (Reasoning + Acting) loop
        self.memory.append({"role": "system", "content": f"You are {self.name}, an autonomous agent. Use available tools to solve the objective."})
        self.memory.append({"role": "user", "content": objective})

        for step in range(1, max_steps + 1):
            print(f"--- [Step {step}] Planning & Decision Phase ---")
            
            # Simulated LLM response deciding whether to call a tool or finish
            llm_decision = self._simulate_llm_reasoning(step, objective)
            
            if llm_decision["action"] == "FINISH":
                print(f" Objective Complete: {llm_decision['final_answer']}")
                return llm_decision['final_answer']

            tool_name = llm_decision["tool"]
            tool_args = llm_decision["args"]
            print(f" Action Chosen: Execute `{tool_name}` with args {tool_args}")

            # Execute tool and catch errors (Self-Correction Loop)
            try:
                tool_result = self.tools[tool_name](**tool_args)
                observation = f"SUCCESS: {tool_result}"
            except Exception as e:
                observation = f"ERROR: Tool execution failed with exception: {str(e)}. Adjust approach."

            print(f" Observation: {observation}\n")
            
            # Feed the observation back into memory so the agent adapts
            self.memory.append({"role": "assistant", "content": json.dumps(llm_decision)})
            self.memory.append({"role": "tool_response", "content": observation})

    def _simulate_llm_reasoning(self, step: int, objective: str) -> Dict[str, Any]:
        """Simulates LLM tool-calling decision making."""
        if step == 1:
            return {"action": "USE_TOOL", "tool": "check_pipeline_logs", "args": {"pipeline_id": "pl_delta_q3"}}
        elif step == 2:
            return {"action": "USE_TOOL", "tool": "optimize_delta_table", "args": {"table_name": "business_monthly", "strategy": "ZORDER"}}
        else:
            return {"action": "FINISH", "final_answer": "Pipeline optimized successfully using Z-Ordering."}

# --- TOOLS DEFINITION ---
def check_pipeline_logs(pipeline_id: str) -> str:
    return f"Logs retrieved for {pipeline_id}: High read-latency on 'business_monthly' Delta table."

def optimize_delta_table(table_name: str, strategy: str) -> str:
    return f"Table '{table_name}' optimized using {strategy}. Query speed improved by 42%."

# --- INITIALIZATION & RUN ---
tools_registry = {
    "check_pipeline_logs": check_pipeline_logs,
    "optimize_delta_table": optimize_delta_table
}

agent = AutonomousAgent("DataOps_Agent", tools_registry)
agent.plan_and_execute("Identify bottleneck in pl_delta_q3 and resolve it.")