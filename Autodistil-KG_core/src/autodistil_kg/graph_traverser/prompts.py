"""
Prompts used by the Graph Traverser Agent.

This module contains all prompts used in the agent, versioned for easy editing
and tracking. Each prompt is versioned (V1, V2, etc.) to allow for evolution
while maintaining backward compatibility.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AlignmentConfig

# ============================================================================
# Semantic Node Selection Prompts
# ============================================================================

SEMANTIC_NODE_SELECTION_PROMPT_V1 = """Given the following candidate nodes from a knowledge graph, select the most semantically relevant one to explore next. Consider the context and relationships.

Candidates:
{candidates_list}

Respond with just the number (1-{max_candidates}) of the selected node."""


SEMANTIC_NODE_SELECTION_PROMPT_V2 = """You are analyzing a knowledge graph traversal. Given the following candidate nodes, select the most semantically relevant and informative node to explore next. Consider:
- The semantic relationships between nodes
- The information value of each node
- The context of the current traversal path

Candidates:
{candidates_list}

Provide only the number (1-{max_candidates}) of your selected node."""


# Current version (default)
SEMANTIC_NODE_SELECTION_PROMPT = SEMANTIC_NODE_SELECTION_PROMPT_V1


# ============================================================================
# Node Context Building Templates
# ============================================================================

NODE_CONTEXT_TEMPLATE_V1 = """Node Information:
Labels: {labels}
Properties: {properties}
{neighbors_info}"""


NODE_CONTEXT_TEMPLATE_V2 = """## Node Details

**Labels:** {labels}

**Properties:**
{properties_formatted}

{neighbors_info}"""


# Current version (default)
NODE_CONTEXT_TEMPLATE = NODE_CONTEXT_TEMPLATE_V1


# ============================================================================
# Deep Reasoning Prompts (REASONING strategy)
# ============================================================================

PATH_REASONING_PROMPT_V1 = """You are a domain expert. Analyze the following structured relationship chain and extract the knowledge it encodes.

## Subject
{center_entity}

## Relationship Chain
{path_description}

## Instructions
Think step-by-step through this chain:
1. **Entity Analysis**: What is each entity? What are its key properties and significance?
2. **Relationship Reasoning**: What does each connection tell us? Why do these entities relate this way?
3. **Derived Knowledge**: What deeper understanding emerges from following this full chain? What implicit facts can be inferred?
4. **Contextual Understanding**: How does this contribute to a broader understanding of the subject?

Provide your reasoning as a structured, detailed analysis. Write as a domain expert — explain the concepts and their significance directly. Do NOT reference graphs, nodes, edges, paths, or database structures in your analysis."""


BATCHED_PATH_REASONING_PROMPT_V1 = """You are a domain expert. Analyze MULTIPLE relationship chains and extract the knowledge each encodes.

## Subject
{center_entity}

## Relationship Chains
{paths_section}

## Instructions
For EACH chain above, provide a clearly separated analysis with the chain number as heading:

For each chain, reason step-by-step:
1. **Entity Analysis**: What is each entity? What are its key properties and significance?
2. **Relationship Reasoning**: What does each connection tell us? Why do these entities relate this way?
3. **Derived Knowledge**: What deeper understanding emerges from following this full chain? What implicit facts can be inferred?
4. **Contextual Understanding**: How does this contribute to a broader understanding of the subject?

Separate each chain analysis with "---PATH_SEPARATOR---" on its own line.

Write as a domain expert — explain the concepts and their significance directly. Do NOT reference graphs, nodes, edges, paths, or database structures in your analysis."""


SUBGRAPH_SYNTHESIS_PROMPT_V1 = """You are a domain expert synthesizing comprehensive knowledge about an entity. Combine the following analyses into a unified, authoritative explanation.

## Subject
{center_entity}

## Analyses
{path_analyses}

## Scope
- Entities covered: {num_nodes}
- Relationships covered: {num_edges}
- Analyses synthesized: {num_paths}

## Instructions
Synthesize all analyses into a comprehensive knowledge summary:
1. **Core Concepts**: What are the fundamental concepts? How do they relate to the subject?
2. **Key Relationships**: What are the most important connections and associations? Are there recurring patterns?
3. **Derived Insights**: What complex facts emerge from combining multiple pieces of evidence? What can we infer that goes beyond individual findings?
4. **Teaching Summary**: Write a clear, detailed explanation that teaches everything important about this entity — its properties, significance, related concepts, and key inferences.

Write as a domain expert in a clear, educational style. Explain concepts and their significance directly — do NOT reference graphs, nodes, edges, paths, databases, or data structures. Use natural language for relationships (e.g., say "is associated with" not "ASSOCIATES_DaG"; say "treats" not "TREATS_CtD")."""


REASONING_QA_GENERATION_PROMPT_V1 = """Based on the following knowledge analysis, generate a high-quality question-answer pair.

## Knowledge Analysis
{synthesis}

## Subject
{center_entity}

## Instructions
Generate a question that:
- Requires connecting multiple pieces of knowledge
- Cannot be answered with a single fact lookup
- Tests understanding of relationships, mechanisms, or reasoning chains
- Sounds like a natural question someone knowledgeable in this domain would ask

Then provide a comprehensive, well-structured answer that:
- Explains the reasoning step by step using proper domain terminology
- References specific entities and concepts by name
- Includes both established facts and logical inferences
- Is detailed enough to teach the concepts

IMPORTANT: Write both question and answer as domain-native content. Do NOT mention knowledge graphs, nodes, edges, paths, traversals, hops, or database structures. Do NOT use internal relationship codes (e.g., ASSOCIATES_DaG, TREATS_CtD, PALLIATES_CpD) — use natural language instead.

Format your response as:
**Question:** [Your question here]

**Answer:** [Your detailed answer here]"""


REASONING_QA_GENERATION_PROMPT_V2 = """Based on the following knowledge analysis, generate a high-quality question-answer pair that is STRICTLY GROUNDED in the provided information.

## Knowledge Analysis
{synthesis}

## Subject
{center_entity}

## Instructions
Generate a question that:
- Requires connecting multiple pieces of knowledge from the analysis above
- Cannot be answered with a single fact lookup
- Tests understanding of relationships, mechanisms, or reasoning chains
- Sounds like a natural question someone knowledgeable in this domain would ask

Then provide a comprehensive, well-structured answer that:
- Uses ONLY information present in the Knowledge Analysis above — do not introduce external facts, statistics, or claims not supported by the analysis
- Explains the reasoning step by step using proper domain terminology
- References specific entities and concepts by name as they appear in the analysis
- Clearly distinguishes between established facts (from the analysis) and logical inferences drawn from those facts
- When making inferences, explicitly signals them (e.g., "This suggests...", "This implies...", "Consequently...")
- Does NOT fabricate specific numbers, percentages, correlation coefficients, p-values, or quantitative data unless they appear in the analysis
- Does NOT invent names of drugs, proteins, genes, or other entities not mentioned in the analysis
- If the analysis does not cover a relevant aspect, omit it rather than fabricating an answer

IMPORTANT: Write both question and answer as domain-native content. Do NOT mention knowledge graphs, nodes, edges, paths, traversals, hops, or database structures. Do NOT use internal relationship codes (e.g., ASSOCIATES_DaG, TREATS_CtD, PALLIATES_CpD) — use natural language instead.

Format your response as:
**Question:** [Your question here]

**Answer:** [Your detailed answer here]"""


SUBGRAPH_SYNTHESIS_PROMPT_V2 = """You are a domain expert synthesizing comprehensive knowledge about an entity. Combine the following analyses into a unified, authoritative explanation that clearly distinguishes between established facts and inferences.

## Subject
{center_entity}

## Analyses
{path_analyses}

## Scope
- Entities covered: {num_nodes}
- Relationships covered: {num_edges}
- Analyses synthesized: {num_paths}

## Instructions
Synthesize all analyses into a comprehensive knowledge summary:
1. **Core Concepts**: What are the fundamental concepts? How do they relate to the subject? State only what is directly supported by the analyses above.
2. **Key Relationships**: What are the most important connections and associations? Are there recurring patterns? For each relationship, use the specific entity names from the analyses.
3. **Derived Insights**: What complex facts emerge from combining multiple pieces of evidence? Clearly mark these as inferences (e.g., "This suggests...", "This implies...") and explain the reasoning chain that supports each inference.
4. **Teaching Summary**: Write a clear, detailed explanation that teaches everything important about this entity — its properties, significance, related concepts, and key inferences.

GROUNDING RULES:
- Only include information that is supported by the analyses above
- Do NOT introduce external facts, statistics, or claims not present in the analyses
- Do NOT fabricate specific numbers, percentages, or quantitative data
- Do NOT invent names of entities (drugs, genes, proteins, diseases) not mentioned in the analyses
- If the analyses contain conflicting information, note the conflict rather than choosing one version
- Clearly separate what is directly stated in the analyses from what you infer

Write as a domain expert in a clear, educational style. Explain concepts and their significance directly — do NOT reference graphs, nodes, edges, paths, databases, or data structures. Use natural language for relationships (e.g., say "is associated with" not "ASSOCIATES_DaG"; say "treats" not "TREATS_CtD")."""


SYNTHESIS_QA_COMBINED_PROMPT_V1 = """You are a domain expert synthesizing comprehensive knowledge and generating educational content.

## Subject
{center_entity}

## Analyses
{path_analyses}

## Scope
- Entities covered: {num_nodes}
- Relationships covered: {num_edges}
- Analyses synthesized: {num_paths}

## Instructions
Complete TWO tasks in a single response:

### Task 1 — Knowledge Synthesis
Synthesize all analyses into a comprehensive knowledge summary covering:
1. **Core Concepts**: Fundamental concepts related to the subject
2. **Key Relationships**: Most important connections, associations, and recurring patterns
3. **Derived Insights**: Complex facts that emerge from combining evidence; inferences beyond individual findings
4. **Teaching Summary**: Clear, detailed explanation teaching everything important about this entity — its properties, significance, related concepts, and practical implications

### Task 2 — Question-Answer Pair
Generate a high-quality question-answer pair from your synthesis:
- The question must require connecting multiple pieces of knowledge (not a single fact lookup)
- The answer must explain reasoning step by step, reference specific entities by name, and include logical inferences

IMPORTANT: Write everything as domain-native content. Do NOT mention knowledge graphs, nodes, edges, paths, traversals, hops, or database structures. Do NOT use internal relationship codes (e.g., ASSOCIATES_DaG, TREATS_CtD, PALLIATES_CpD) — use natural language instead (e.g., "is associated with", "treats", "palliates").

Format your FULL response EXACTLY as:

## Synthesis
[Your comprehensive knowledge synthesis here]

## QA
**Question:** [Your question here]

**Answer:** [Your detailed answer here]"""


SYNTHESIS_QA_COMBINED_PROMPT_V2 = """You are a domain expert synthesizing comprehensive knowledge and generating educational content that is STRICTLY GROUNDED in the provided analyses.

## Subject
{center_entity}

## Analyses
{path_analyses}

## Scope
- Entities covered: {num_nodes}
- Relationships covered: {num_edges}
- Analyses synthesized: {num_paths}

## Instructions
Complete TWO tasks in a single response:

### Task 1 — Knowledge Synthesis
Synthesize all analyses into a comprehensive knowledge summary covering:
1. **Core Concepts**: Fundamental concepts related to the subject, using only information from the analyses
2. **Key Relationships**: Most important connections, associations, and recurring patterns — reference specific entity names from the analyses
3. **Derived Insights**: Complex facts that emerge from combining evidence; clearly mark inferences with language like "This suggests..." or "This implies..." and explain the reasoning chain
4. **Teaching Summary**: Clear, detailed explanation teaching everything important about this entity

### Task 2 — Question-Answer Pair
Generate a high-quality question-answer pair from your synthesis:
- The question must require connecting multiple pieces of knowledge (not a single fact lookup)
- The answer must explain reasoning step by step, reference specific entities by name, and include logical inferences
- The answer must ONLY use information present in the synthesis — do not introduce external facts
- Clearly distinguish between established facts and inferences in the answer
- Do NOT fabricate specific numbers, percentages, correlation coefficients, or quantitative data unless they appear in the analyses
- Do NOT invent names of entities (drugs, genes, proteins, diseases) not mentioned in the analyses

IMPORTANT: Write everything as domain-native content. Do NOT mention knowledge graphs, nodes, edges, paths, traversals, hops, or database structures. Do NOT use internal relationship codes (e.g., ASSOCIATES_DaG, TREATS_CtD, PALLIATES_CpD) — use natural language instead (e.g., "is associated with", "treats", "palliates").

Format your FULL response EXACTLY as:

## Synthesis
[Your comprehensive knowledge synthesis here]

## QA
**Question:** [Your question here]

**Answer:** [Your detailed answer here]"""


# ============================================================================
# Quality Gate Prompt (post-generation scoring)
# ============================================================================

QUALITY_SCORING_PROMPT_V1 = """You are a training-data quality reviewer. Score the following question-answer pair on three dimensions. Each dimension is 0–10.

## Q&A Pair
**Question:** {question}

**Answer:** {answer}
{reference_section}
## Dimensions
1. **Relevance** — Does the answer directly and fully address the question?
2. **Groundedness** — Is every claim in the answer supported by the knowledge analysis or reference material (if provided)? Penalise fabricated facts, invented entities, or unsupported numbers.
3. **Completeness** — Does the answer cover all major aspects a domain expert would expect?

Respond with EXACTLY three lines, no commentary:
relevance: <0-10>
groundedness: <0-10>
completeness: <0-10>"""


# Current versions
PATH_REASONING_PROMPT = PATH_REASONING_PROMPT_V1
SUBGRAPH_SYNTHESIS_PROMPT = SUBGRAPH_SYNTHESIS_PROMPT_V2
REASONING_QA_GENERATION_PROMPT = REASONING_QA_GENERATION_PROMPT_V2
SYNTHESIS_QA_COMBINED_PROMPT = SYNTHESIS_QA_COMBINED_PROMPT_V2
QUALITY_SCORING_PROMPT = QUALITY_SCORING_PROMPT_V1


# ============================================================================
# Helper Functions
# ============================================================================

def build_alignment_block(
    alignment: "AlignmentConfig | None" = None,
    reference_excerpts: str = "",
) -> str:
    """Build an optional alignment context block to inject into prompts.

    Returns an empty string when no alignment is configured, so callers can
    unconditionally append it without conditional logic.
    """
    if alignment is None:
        return ""

    sections: list[str] = []

    # Domain focus
    if alignment.domain_focus:
        sections.append(f"**Domain focus:** {alignment.domain_focus}")
    if alignment.domain_keywords:
        sections.append(f"**Domain keywords:** {', '.join(alignment.domain_keywords)}")

    # Style / audience
    if alignment.target_audience:
        sections.append(f"**Target audience:** {alignment.target_audience}")
    if alignment.style_guide:
        sections.append(f"**Style guide:** {alignment.style_guide}")
    length_parts: list[str] = []
    if alignment.min_answer_length:
        length_parts.append(f"at least {alignment.min_answer_length} words")
    if alignment.max_answer_length:
        length_parts.append(f"at most {alignment.max_answer_length} words")
    if length_parts:
        sections.append(f"**Answer length:** {', '.join(length_parts)}")

    # Reference material
    if reference_excerpts:
        sections.append(
            "**Reference material** (use to ground your output — prefer "
            "information that aligns with these passages):\n"
            f"{reference_excerpts}"
        )

    if not sections:
        return ""
    body = "\n".join(sections)
    return f"\n\n## Alignment Context\n{body}\n"


def format_quality_scoring_prompt(
    question: str,
    answer: str,
    reference: str = "",
) -> str:
    """Build the quality-gate scoring prompt."""
    ref_section = ""
    if reference:
        ref_section = f"\n## Reference Material\n{reference}\n"
    return QUALITY_SCORING_PROMPT.format(
        question=question,
        answer=answer,
        reference_section=ref_section,
    )


def format_path_description(path: list) -> str:
    """
    Format a subgraph path into a human-readable chain description.

    Args:
        path: Alternating list of node dicts and edge dicts.
              Nodes have 'id', 'labels', 'properties'.
              Edges have 'source_id', 'target_id', 'type', 'properties'.

    Returns:
        A readable string like:
        [Person: Alice] --(KNOWS)--> [Person: Bob] --(WORKS_AT)--> [Company: Acme]
    """
    parts = []
    for i, element in enumerate(path):
        if i % 2 == 0:
            # Node
            labels = ", ".join(element.get("labels", []))
            name = (
                element.get("properties", {}).get("name")
                or element.get("properties", {}).get("title")
                or element.get("properties", {}).get("id")
                or element.get("id", "?")
            )
            props = element.get("properties", {})
            prop_summary = ", ".join(
                f"{k}: {v}" for k, v in list(props.items())[:5]
            )
            parts.append(f"[{labels}: {name}]({prop_summary})")
        else:
            # Edge
            rel_type = element.get("type", "RELATED_TO")
            rel_props = element.get("properties", {})
            if rel_props:
                prop_str = " {" + ", ".join(f"{k}: {v}" for k, v in list(rel_props.items())[:3]) + "}"
            else:
                prop_str = ""
            parts.append(f"--({rel_type}{prop_str})-->")
    return " ".join(parts)


def format_center_entity(node: dict) -> str:
    """Format a center node for inclusion in prompts."""
    labels = ", ".join(node.get("labels", []))
    props = node.get("properties", {})
    prop_lines = "\n".join(f"  - {k}: {v}" for k, v in props.items())
    name = props.get("name") or props.get("title") or node.get("id", "Unknown")
    return f"**{name}** (Labels: {labels})\nProperties:\n{prop_lines}"


def format_path_reasoning_prompt(
    center_node: dict,
    path: list,
    version: str = "V1",
    alignment: "AlignmentConfig | None" = None,
    reference_excerpts: str = "",
) -> str:
    """
    Build a prompt asking the LLM to reason through a single path.

    Args:
        center_node: The center entity dict.
        path: Alternating node/edge list.
        version: Prompt version.
        alignment: Optional alignment configuration.
        reference_excerpts: Optional reference text for grounding.

    Returns:
        Formatted prompt string.
    """
    template = PATH_REASONING_PROMPT_V1
    prompt = template.format(
        center_entity=format_center_entity(center_node),
        path_description=format_path_description(path),
    )
    return prompt + build_alignment_block(alignment, reference_excerpts)


def format_batched_path_reasoning_prompt(
    center_node: dict,
    paths: list,
    alignment: "AlignmentConfig | None" = None,
    reference_excerpts: str = "",
) -> str:
    """
    Build a prompt asking the LLM to reason through multiple paths at once.

    Args:
        center_node: The center entity dict.
        paths: List of paths (each an alternating node/edge list).
        alignment: Optional alignment configuration.
        reference_excerpts: Optional reference text for grounding.

    Returns:
        Formatted prompt string.
    """
    paths_section = ""
    for i, path in enumerate(paths, 1):
        paths_section += f"\n### Path {i}\n{format_path_description(path)}\n"

    prompt = BATCHED_PATH_REASONING_PROMPT_V1.format(
        center_entity=format_center_entity(center_node),
        paths_section=paths_section,
    )
    return prompt + build_alignment_block(alignment, reference_excerpts)


def format_subgraph_synthesis_prompt(
    center_node: dict,
    path_analyses: list,
    num_nodes: int,
    num_edges: int,
    version: str = "V1",
    alignment: "AlignmentConfig | None" = None,
    reference_excerpts: str = "",
) -> str:
    """
    Build a prompt to synthesize multiple path analyses.

    Args:
        center_node: Center entity dict.
        path_analyses: List of strings, each a path-level reasoning.
        num_nodes: Total unique nodes in the subgraph.
        num_edges: Total edges in the subgraph.
        version: Prompt version.
        alignment: Optional alignment configuration.
        reference_excerpts: Optional reference text for grounding.

    Returns:
        Formatted prompt string.
    """
    analyses_text = ""
    for i, analysis in enumerate(path_analyses, 1):
        analyses_text += f"\n### Path {i}\n{analysis}\n"

    if version.upper() == "V2":
        template = SUBGRAPH_SYNTHESIS_PROMPT_V2
    else:
        template = SUBGRAPH_SYNTHESIS_PROMPT_V1
    prompt = template.format(
        center_entity=format_center_entity(center_node),
        path_analyses=analyses_text,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_paths=len(path_analyses),
    )
    return prompt + build_alignment_block(alignment, reference_excerpts)


def format_reasoning_qa_prompt(
    center_node: dict,
    synthesis: str,
    version: str = "V1",
    alignment: "AlignmentConfig | None" = None,
    reference_excerpts: str = "",
) -> str:
    """
    Build a prompt to generate a distillation-ready QA pair from synthesis.

    Args:
        center_node: Center entity dict.
        synthesis: The synthesized knowledge text.
        version: Prompt version.
        alignment: Optional alignment configuration.
        reference_excerpts: Optional reference text for grounding.

    Returns:
        Formatted prompt string.
    """
    if version.upper() == "V2":
        template = REASONING_QA_GENERATION_PROMPT_V2
    else:
        template = REASONING_QA_GENERATION_PROMPT_V1
    prompt = template.format(
        center_entity=format_center_entity(center_node),
        synthesis=synthesis,
    )
    return prompt + build_alignment_block(alignment, reference_excerpts)


def format_synthesis_qa_combined_prompt(
    center_node: dict,
    path_analyses: list,
    num_nodes: int,
    num_edges: int,
    alignment: "AlignmentConfig | None" = None,
    reference_excerpts: str = "",
) -> str:
    """
    Build a single prompt that synthesizes path analyses AND generates a QA pair.

    This replaces the two separate calls (synthesis + QA generation) with one
    combined call, halving the LLM calls for this stage.

    Args:
        center_node: Center entity dict.
        path_analyses: List of strings, each a path-level reasoning.
        num_nodes: Total unique nodes in the subgraph.
        num_edges: Total edges in the subgraph.
        alignment: Optional alignment configuration.
        reference_excerpts: Optional reference text for grounding.

    Returns:
        Formatted prompt string.
    """
    analyses_text = ""
    for i, analysis in enumerate(path_analyses, 1):
        analyses_text += f"\n### Path {i}\n{analysis}\n"

    prompt = SYNTHESIS_QA_COMBINED_PROMPT_V1.format(
        center_entity=format_center_entity(center_node),
        path_analyses=analyses_text,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_paths=len(path_analyses),
    )
    return prompt + build_alignment_block(alignment, reference_excerpts)


def format_semantic_selection_prompt(
    candidate_info: list,
    version: str = "V1",
    alignment: AlignmentConfig | None = None,
) -> str:
    """
    Format the semantic node selection prompt with candidate information.

    Args:
        candidate_info: List of dicts with 'id', 'labels', 'properties'
        version: Prompt version to use ('V1', 'V2', or 'current')
        alignment: Optional alignment config for domain biasing.

    Returns:
        Formatted prompt string
    """
    if version == "current" or version.upper() == "V1":
        prompt_template = SEMANTIC_NODE_SELECTION_PROMPT_V1
    elif version.upper() == "V2":
        prompt_template = SEMANTIC_NODE_SELECTION_PROMPT_V2
    else:
        prompt_template = SEMANTIC_NODE_SELECTION_PROMPT

    # Format candidates list
    candidates_list = []
    for i, info in enumerate(candidate_info, start=1):
        labels_str = ", ".join(info.get("labels", []))
        props_str = str(info.get("properties", {}))
        candidates_list.append(
            f"{i}. Node {info['id']}: {labels_str} - {props_str}"
        )

    candidates_text = "\n".join(candidates_list)
    max_candidates = len(candidate_info)

    prompt = prompt_template.format(
        candidates_list=candidates_text,
        max_candidates=max_candidates,
    )

    # Inject domain bias into semantic selection
    if alignment and (alignment.domain_focus or alignment.domain_keywords):
        bias_parts: list[str] = []
        if alignment.domain_focus:
            bias_parts.append(f"Prefer nodes most relevant to: {alignment.domain_focus}")
        if alignment.domain_keywords:
            bias_parts.append(f"Prioritise nodes related to: {', '.join(alignment.domain_keywords)}")
        prompt += "\n\n" + "\n".join(bias_parts)

    return prompt


def format_node_context(
    labels: list,
    properties: dict,
    neighbors_count: int = 0,
    version: str = "V1"
) -> str:
    """
    Format node context information for LLM prompts.
    
    Args:
        labels: List of node labels
        properties: Dictionary of node properties
        neighbors_count: Number of neighboring nodes
        version: Template version to use ('V1', 'V2', or 'current')
    
    Returns:
        Formatted context string
    """
    if version == "current" or version.upper() == "V1":
        template = NODE_CONTEXT_TEMPLATE_V1
        labels_str = ", ".join(labels) if labels else "None"
        props_str = str(properties)
        neighbors_info = f"\nRelated Nodes: {neighbors_count} neighbors" if neighbors_count > 0 else ""
        
        return template.format(
            labels=labels_str,
            properties=props_str,
            neighbors_info=neighbors_info
        )
    elif version.upper() == "V2":
        template = NODE_CONTEXT_TEMPLATE_V2
        labels_str = ", ".join(labels) if labels else "None"
        
        # Format properties as key-value pairs
        props_lines = []
        for key, value in properties.items():
            props_lines.append(f"  - {key}: {value}")
        props_formatted = "\n".join(props_lines) if props_lines else "  (no properties)"
        
        neighbors_info = f"\n**Related Nodes:** {neighbors_count} neighbors" if neighbors_count > 0 else ""
        
        return template.format(
            labels=labels_str,
            properties_formatted=props_formatted,
            neighbors_info=neighbors_info
        )
    else:
        # Default to V1
        return format_node_context(labels, properties, neighbors_count, "V1")
