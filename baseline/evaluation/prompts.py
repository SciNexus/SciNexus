EFFECTIVENESS_PROMPT = 
"""You are an expert scientific reviewer. Your task is to evaluate the **Effectiveness** of a newly generated research hypothesis.
**Background Context:**
- Scientific Context: {scientific_context}
**Generated Hypothesis:**
{generated_hypothesis}
**Evaluation Criteria:**
Effectiveness measures the semantic alignment between the generated hypothesis and the target research objectives (i.e., resolving the stated limitation). A highly effective hypothesis should provide a logical, targeted, and valid scientific approach to overcoming the limitation of the current method.
**Instructions:**
Carefully analyze the logical connection between the limitation and the proposed hypothesis. Does the hypothesis effectively address the problem?
Provide a brief step-by-step reasoning (max 3 sentences), and then output your final decision on a new line strictly as:
"VOTE: 1" (if it is effective and well-aligned) OR "VOTE: 0" (if it is ineffective or misaligned)."""
EFFECTIVENESS_PROMPT_BACKWARD = """You are an expert scientific reviewer. Your task is to evaluate the **Effectiveness** of a newly generated research hypothesis.
**Generated Hypothesis:**
{generated_hypothesis}
**Background Context:**
- Scientific Context: {scientific_context}
**Evaluation Criteria:**
Effectiveness measures the semantic alignment between the generated hypothesis and the target research objectives (i.e., resolving the stated limitation). A highly effective hypothesis should provide a logical, targeted, and valid scientific approach to overcoming the limitation of the current method.
**Instructions:**
Carefully analyze the logical connection between the limitation and the proposed hypothesis. Does the hypothesis effectively address the problem?
Provide a brief step-by-step reasoning (max 3 sentences), and then output your final decision on a new line strictly as:
"VOTE: 1" (if it is effective and well-aligned) OR "VOTE: 0" (if it is ineffective or misaligned)."""
NOVELTY_PROMPT = """You are an expert scientific reviewer. Your task is to evaluate the **Novelty** of a newly generated research hypothesis.
**Background Context (Existing Literature):**
- Scientific Context: {scientific_context}
**Generated Hypothesis:**
{generated_hypothesis}
**Evaluation Criteria:**
Novelty quantifies the originality of the hypothesis. It should reward conceptual divergence from existing literature rather than the memorization or trivial modification of prior works. A highly novel hypothesis introduces a genuinely new concept, mechanism, paradigm, or perspective to address the limitation.
**Instructions:**
Assess the conceptual divergence of the generated hypothesis compared to the current method. Is it a significant scientific leap or just an incremental tweak?
Provide a brief step-by-step reasoning (max 3 sentences), and then output your final decision on a new line strictly as:
"VOTE: 1" (if it is highly novel and divergent) OR "VOTE: 0" (if it is trivial, incremental, or lacks originality)."""
NOVELTY_PROMPT_BACKWARD = """You are an expert scientific reviewer. Your task is to evaluate the **Novelty** of a newly generated research hypothesis.
**Generated Hypothesis:**
{generated_hypothesis}
**Background Context (Existing Literature):**
- Scientific Context: {scientific_context}
**Evaluation Criteria:**
Novelty quantifies the originality of the hypothesis. It should reward conceptual divergence from existing literature rather than the memorization or trivial modification of prior works. A highly novel hypothesis introduces a genuinely new concept, mechanism, paradigm, or perspective to address the limitation.
**Instructions:**
Assess the conceptual divergence of the generated hypothesis compared to the current method. Is it a significant scientific leap or just an incremental tweak?
Provide a brief step-by-step reasoning (max 3 sentences), and then output your final decision on a new line strictly as:
"VOTE: 1" (if it is highly novel and divergent) OR "VOTE: 0" (if it is trivial, incremental, or lacks originality)."""
WIN_RATE_PROMPT = """You are an expert scientific reviewer. Your task is to perform a pairwise evaluation between two proposed research texts to determine which one presents a superior scientific concept.
**Background Context:**
- Scientific Context: {scientific_context}
**Candidate A:**
{text_A}
**Candidate B:**
{text_B}
**Evaluation Criteria:**
Evaluate both candidates based on the following three dimensions:
1. Novelty: Which text is more original and conceptually divergent from the current method?
**Instructions:**
Compare Candidate A and Candidate B comprehensively. Do not let the length of the text bias your decision.
Provide a brief comparative reasoning (max 4 sentences), and then output your final choice on a new line strictly as:
"VOTE: A" (if Candidate A is superior) OR "VOTE: B" (if Candidate B is superior)."""
