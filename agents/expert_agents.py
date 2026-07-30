from __future__ import annotations
import json
from SciNexus.config import ModelConfig
from ..hypothesis_text import normalize_hypothesis_text
from .base_agent import BaseAgent
ROLE_DEFINITIONS: dict[str, dict] = {
    "Life Scientist": {
        "full_name": "Life Scientist",
        "description": (
            "You are Life Scientist, assessing the complexity of biological systems, evolutionary dynamics, and living mechanisms. "
        ),
    },
    "Chemist": {
        "full_name": "Chemist",
        "description": (
            "You are The Computational Systems Architect, assessing algorithmic logic, computational complexity, "
            "and information flow. "
        ),
    },
    "Computer Scientist": {
        "full_name": "Computer Scientist",
        "description": (
            "You are Computer Scientist, assessing algorithmic logic, computational complexity, and information flow."
        ),
    },
    "Mathematician": {
        "full_name": "Mathematician",
        "description": (
            "You are Mathematician, distilling complex, multidisciplinary phenomena into rigorous structures."
        ),
    },
    "Physicist": {
        "full_name": "Physicist",
        "description": (
            "You are Physicist, grounding multidisciplinary ideas in the fundamental laws of nature, thermodynamics, and quantum mechanics. "
        ),
    },
    "Earth Scientist": {
        "full_name": "Earth Scientist",
        "description": (
            "You are Earth Scientist,exploring geological processes, climate dynamics, and biogeochemical cycles. "
        ),
    },
    "Materials Scientist": {
        "full_name": "Materials Scientist",
        "description": (
            "You are Materials Scientist, exploring structure-property relationships, phase transitions, and novel material synthesis.  "
        ),
    },
}
INTERACTION_SYSTEM: dict[str, str] = {
    "debate": (
        "You are engaging in a cross-disciplinary scientific DEBATE. Structure your response as follows in JSON format:\n"
        "{'analysis': 'Critically deconstruct the incoming hypothesis using your specific domain expertise. "
        "Explicitly expose its theoretical blind spots, hidden assumptions, or interdisciplinary frictions. "
        "Be rigorous and cite concrete principles from your field.', "
        "'hypothesis': 'Propose a more robust, alternative hypothesis that directly resolves these structural flaws "
        "while preserving the valid insights and innovative core of the original idea.'}"
    ),
    "analogy": (
        "You are drawing a cross-disciplinary scientific ANALOGY. Structure your response as follows in JSON format:\n"
        "{'analysis': 'Identify a proven mechanism, model, or solved problem from your domain that is structurally "
        "isomorphic to the incoming hypothesis. Explicitly map the abstract components to demonstrate the parallel.', "
        "'hypothesis': 'Leverage this isomorphic mapping to formulate a novel, richer hypothesis. Import established "
        "mathematical, physical, or biological solutions from your domain to unlock new pathways in the target context.'}"
    ),
    "review": (
        "You are conducting a rigorous scientific REVIEW. Structure your response as follows in JSON format:\n"
        "{'analysis': 'Evaluate the incoming hypothesis for feasibility, logical soundness, and scientific completeness "
        "exclusively through your expert lens. Identify the single most critical theoretical gap, scaling limit, or empirical bottleneck.', "
        "'hypothesis': 'Synthesize a refined hypothesis that bridges this multidisciplinary gap. Enhance its experimental "
        "or computational testability, and elevate the overall rigor of the scientific design.'}"
    ),
    "clarification": (
        "You are seeking scientific CLARIFICATION. Structure your response as follows in JSON format:\n"
        "{'analysis': 'Identify the conceptual ambiguities, undefined variables, or missing domain-specific parameters "
        "in the incoming hypothesis that create friction or make it unactionable within your field.', "
        "'hypothesis': 'Resolve these interdisciplinary translational issues using your strict domain standards. "
        "Output a precise, logically well-scoped, and empirically measurable refined hypothesis.'}"
    )
}
HYPOTHESIS_WRITING_GUIDE = """\
Please leverage your professional expertise to address the limitations within the aforementioned scientific context, relying on datasets. The output hypothesis includes the following modules:
---
(1) Clearly define the core problem the methodology targets and specify the expected quantitative or qualitative outcomes of its successful application.
(2) Theoretical Anchor & Deductive Logic: Elucidate the underlying scientific axioms or prior assumptions, and explain the intrinsic logical mechanisms that allow the execution steps to synergize effectively.
(3) Structural Operational Flow: Deconstruct the abstract methodology into specific, independent executable modules, and clearly map the linear or closed-loop transmission pathways of data or information within the system.
(4) Boundary Conditions & Failure Modes: Proactively demarcate the physical conditions or data scale prerequisites for application, and explicitly identify extreme scenarios that would lead to system degradation or inference failure.
(5) Observable Evaluation Metrics: Establish specific quantitative metrics to measure the methodology's efficacy, and set industry or traditional baselines for scientific comparison.
(6) Falsifiability Criteria: Pre-define clear scientific falsification criteria: identifying exactly what specific data validation outcomes would conclusively prove the methodological hypothesis invalid or failed.  """
class ExpertAgent(BaseAgent):
    def __init__(self, role: str, config: ModelConfig):
        super().__init__(role, config)
        if role not in ROLE_DEFINITIONS:
            raise ValueError(
                f"Unknown role '{role}'. Available: {list(ROLE_DEFINITIONS.keys())}"
            )
        self.role_def = ROLE_DEFINITIONS[role]
        self.current_hypothesis = ""
        self.hypothesis_history: list[str] = []
    def generate_initial_hypothesis(self, task: dict) -> str:
        system_prompt = (
            f"{self.role_def['description']}\n\n"
            "You will receive a scientific context. "
            "Generate a novel, specific hypothesis based on your expertise and the scientific context. "
            "Keep the entire response concise and tightly focused.\n\n"
            f"{HYPOTHESIS_WRITING_GUIDE}\n\n"
            "Respond ONLY with a valid JSON object. The `hypothesis` field must contain the six headings above."
            "required headings:\n"
            '{"hypothesis": "..."}'
        )
        user_prompt = (
            f"Scientific Problem\n"
            f"──────────────────\n"
            f"Scientific Context: {task}\n\n"
            f"As {self.role_def['full_name']}, generate your initial hypothesis."
        )
        result = self.call_llm_json(system_prompt, user_prompt, max_tokens=4096)
        hypothesis = normalize_hypothesis_text(
            result.get("hypothesis") or result.get("raw_response", "")
        )
        self._update_hypothesis(hypothesis)
        return hypothesis
    def respond_to_interaction(
        self,
        task: dict,
        src_role: str,
        src_message: str,
        interaction_type: str,
    ) -> str:
        interaction_preamble = INTERACTION_SYSTEM.get(interaction_type, "")
        history_summary = self._build_history_summary()
        system_prompt = (
            f"{self.role_def['description']}\n\n"
            "Keep the entire response concise and tightly focused.\n\n"
            "Respond ONLY with a valid JSON object:\n"
            '{"analysis": "...", "hypothesis": "..."}\n'
            f"{HYPOTHESIS_WRITING_GUIDE}\n\n"
            "The `analysis` field covers your interaction-specific critique or reasoning. "
            "The `hypothesis` field must follow the four required headings above."
        )
        user_prompt = (
            f"Original Scientific Problem\n"
            f"───────────────────────────\n"
            f"Scientific Context: {task}\n\n"
            f"Your prior hypothesis : {self.current_hypothesis}\n"
            f"Your history summary  : {history_summary}\n\n"
            f"You received a Hypothesis from other expert:\n"
            f"{src_message}\n\n"
            f"As {self.role_def['full_name']},{interaction_preamble}, respond and produce an analysis and improved hypothesis. "
            "Respond ONLY with a valid JSON object:\n"
            '{"analysis":..., "hypothesis":...}'
        )
        result = self.call_llm_json(system_prompt, user_prompt, max_tokens=4096)
        try:
            analysis = result.get("analysis")
            hypothesis1 = result.get("hypothesis")
        except Exception:
            data = str(result.get("raw_response", "{}")).replace("```json", "").replace("```", "")
            data = json.loads(data)
            analysis = data.get("analysis")
            hypothesis1 = data.get("hypothesis")
        hypothesis = (
            "analysis: " + normalize_hypothesis_text(analysis)
            + ",hypothesis: " + normalize_hypothesis_text(hypothesis1)
        )
        self._update_hypothesis(hypothesis)
        return hypothesis
    def _update_hypothesis(self, hypothesis) -> None:
        text = normalize_hypothesis_text(hypothesis)
        self.current_hypothesis = text
        self.hypothesis_history.append(text)
    def _build_history_summary(self) -> str:
        if not self.hypothesis_history:
            return "No prior hypotheses."
        if len(self.hypothesis_history) <= 2:
            return " → ".join(self.hypothesis_history[-2:])
        return (
            f"[{len(self.hypothesis_history)} iterations. "
            f"Latest: {self.hypothesis_history[-1]}…]"
        )
    def reset(self) -> None:
        self.current_hypothesis = ""
        self.hypothesis_history = []
def create_default_agents(config: ModelConfig) -> dict[str, ExpertAgent]:
    return {role: ExpertAgent(role, config) for role in config.agent_roles}
