"""
Prompts for the research pipeline.

Single-agent prompts: understanding, planning, extraction, gap detection,
verification, report writing.

Multi-agent prompts: orchestration, per-worker extraction and summary,
synthesis, multi-source report writing.
"""

# ---------------------------------------------------------------------------
# Understanding
# ---------------------------------------------------------------------------

UNDERSTAND_PROMPT = """You are a research assistant. Analyze this query to determine if it is clear enough to research effectively.

USER'S QUERY:
{query}

STEP 1 - Identify what the user is asking about.
STEP 2 - Check for ambiguity across ALL of these dimensions:
  a) POLYSEMY: Could key terms refer to different things? (e.g. "Python" → language vs snake vs movie)
  b) SCOPE: Is the query too broad to research in one pass? (e.g. "Tell me about AI")
  c) TIME PERIOD: Does the answer depend on when? Is a specific time frame needed?
  d) GEOGRAPHY: Does the answer vary by location/country/region?
  e) PERSPECTIVE: Could the user want different angles? (technical vs business vs beginner)
  f) SPECIFICITY: Are there sub-topics and the user likely cares about only some?
  g) COMPARISON BASELINE: If comparing, are the comparands clear?

STEP 3 - A query is CLEAR if a reasonable researcher could proceed without guessing the user's intent.
         A query is AMBIGUOUS if proceeding would risk researching the wrong thing.

Respond with JSON:
{{
    "understanding": "What you understand the user wants (1-2 sentences)",
    "is_clear": true/false,
    "ambiguity_type": "polysemy|scope|time_period|geography|perspective|specificity|comparison|none",
    "ambiguity_reason": "Why it's ambiguous (only if is_clear is false)",
    "clarification_question": "A specific question to resolve the ambiguity (only if is_clear is false)",
    "clarification_options": [
        {{"label": "Short option text", "description": "1-sentence explanation of what this option means and what the research would focus on"}},
        {{"label": "Short option text", "description": "1-sentence explanation"}},
        {{"label": "Short option text", "description": "1-sentence explanation"}}
    ]
}}

GUIDELINES FOR CLARIFICATION OPTIONS:
- Generate 3-5 options that cover the most likely user intents
- Each option should be MEANINGFULLY DIFFERENT (not overlapping)
- Order from most likely to least likely intent
- The label should be concise (3-8 words)
- The description should explain what research would focus on
- If the ambiguity is about scope, offer specific sub-topics
- If about time period, offer specific ranges
- Always make the options mutually exclusive enough that picking one gives clear research direction
"""

# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

PLAN_PROMPT = """You are a research planner. Create a research plan for this query.

USER'S ORIGINAL QUERY:
{query}

{clarification_section}

Think about:
1. What are the key questions that need to be answered?
2. What aspects or dimensions should be covered?
3. What search queries will find the needed information?

Respond with JSON:
{{
    "research_questions": [
        "Key question 1 that needs to be answered",
        "Key question 2 that needs to be answered",
        "Key question 3 that needs to be answered"
    ],
    "aspects_to_cover": [
        "Aspect/dimension 1 to research",
        "Aspect/dimension 2 to research"
    ],
    "initial_searches": [
        "search query 1",
        "search query 2",
        "search query 3",
        "search query 4"
    ]
}}

Generate 3-5 research questions and 4-6 search queries that will comprehensively cover the topic.
"""

# ---------------------------------------------------------------------------
# Evidence extraction (single-agent)
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """Extract key facts from this content relevant to the research.

RESEARCH CONTEXT:
{context}

CONTENT FROM {url}:
{content}

Extract specific, factual information. Each fact should be:
- Specific and verifiable (not vague)
- Directly relevant to the research
- Self-contained (understandable on its own)

Respond with JSON array:
[
    {{
        "fact": "The specific factual claim",
        "quote": "Direct quote or paraphrase from the source",
        "confidence": 0.0-1.0
    }}
]

Return [] if no relevant facts found. Be selective - quality over quantity.
"""

# ---------------------------------------------------------------------------
# Gap detection (single-agent)
# ---------------------------------------------------------------------------

GAP_DETECTION_PROMPT = """You are assessing research completeness.

ORIGINAL QUERY:
{query}
{clarification_section}

RESEARCH QUESTIONS TO ANSWER:
{questions}

EVIDENCE GATHERED SO FAR:
{evidence}

Analyze the coverage:
1. Which research questions are well-answered with strong evidence?
2. Which questions have weak or no evidence?
3. What important aspects are missing?

Respond with JSON:
{{
    "coverage_assessment": [
        {{"question": "question text", "status": "well_covered|partial|missing", "evidence_count": N}},
        ...
    ],
    "overall_coverage": 0.0-1.0,
    "gaps": [
        "Specific gap or missing information 1",
        "Specific gap or missing information 2"
    ],
    "suggested_searches": [
        "search query to fill gap 1",
        "search query to fill gap 2"
    ],
    "ready_to_write": true/false,
    "reasoning": "Why ready or not ready"
}}

Be honest about gaps. It's better to do another search iteration than write an incomplete report.
"""

# ---------------------------------------------------------------------------
# Verification (shared by both paths)
# ---------------------------------------------------------------------------

VERIFY_PROMPT = """You are a fact-checker. Verify claims against the evidence.

CLAIMS TO VERIFY:
{claims}

AVAILABLE EVIDENCE:
{evidence}

For each claim, check if it's supported by the evidence.

Respond with JSON array:
[
    {{
        "claim": "The claim text",
        "supported": true/false,
        "supporting_evidence": [1, 3, 5],
        "confidence": 0.0-1.0,
        "notes": "Any caveats or qualifications"
    }}
]
"""

# ---------------------------------------------------------------------------
# Report writing (single-agent)
# ---------------------------------------------------------------------------

WRITE_PROMPT = """Write a comprehensive research report answering this query.

QUERY:
{query}
{clarification_section}

VERIFIED EVIDENCE (use these as your ONLY source of facts):
{evidence}

SOURCES:
{sources}

{structure_instruction}

IMPORTANT RULES:
1. ONLY include information that appears in the evidence above
2. EVERY factual claim must have a citation [1], [2], etc.
3. If evidence is contradictory, acknowledge both perspectives
4. If evidence is insufficient for some aspect, acknowledge the limitation
5. Be comprehensive but don't make things up

Write in a professional, informative tone.
"""

# Structure instructions injected based on report_structure setting
STRUCTURE_DETAILED = """Structure your report with:
- A clear introduction summarizing the key findings
- Multiple sections, each addressing a different aspect in depth
- Sub-sections where appropriate for complex topics
- Inline citations for every factual claim
- A conclusion synthesizing the main takeaways
- A Sources section at the end"""

STRUCTURE_CONCISE = """Structure your report as a concise briefing:
- A 2-3 sentence executive summary at the top
- Short paragraphs covering the key points (no more than 3-4 paragraphs)
- Inline citations for every factual claim
- A Sources section at the end
Keep it under 500 words. Prioritize the most important findings."""

STRUCTURE_BULLET_POINTS = """Structure your report as a bullet-point summary:
- Start with a 1-2 sentence overview
- Use bullet points for all key findings (grouped by theme)
- Each bullet should be a single clear statement with a citation
- End with a Sources section
No long paragraphs. Every bullet must cite its source."""

def get_structure_instruction(report_structure: str) -> str:
    """Return the formatting instruction for the chosen report structure."""
    return {
        "concise": STRUCTURE_CONCISE,
        "bullet_points": STRUCTURE_BULLET_POINTS,
    }.get(report_structure, STRUCTURE_DETAILED)

# ---------------------------------------------------------------------------
# Orchestration (multi-agent)
# ---------------------------------------------------------------------------

ORCHESTRATE_PROMPT = """You are a research orchestrator. Break down this research query into independent sub-questions that can be researched in parallel.

QUERY: {query}
{clarification_section}

Rules for creating sub-questions:
1. Each sub-question should be INDEPENDENT (can be researched without the others)
2. Together they should FULLY COVER the original query
3. Create 3-5 sub-questions (not more)
4. Each should be specific and searchable

Respond with JSON:
{{
    "sub_questions": [
        {{
            "id": "Q1",
            "question": "Specific sub-question",
            "focus": "What aspect this covers",
            "search_queries": ["search 1", "search 2"]
        }}
    ],
    "reasoning": "Why this breakdown covers the query well"
}}
"""

# ---------------------------------------------------------------------------
# Worker extraction & summary (multi-agent)
# ---------------------------------------------------------------------------

WORKER_EXTRACT_PROMPT = """You are a research worker focused on ONE specific question.

YOUR ASSIGNED QUESTION:
{question}

CONTENT FROM {url}:
{content}

Extract facts ONLY relevant to your assigned question. Be focused.

Respond with JSON array:
[
    {{"fact": "specific fact", "quote": "supporting quote", "confidence": 0.9}}
]

Return [] if nothing relevant to YOUR question.
"""

WORKER_SUMMARY_PROMPT = """Summarize your research findings for this specific question.

YOUR QUESTION:
{question}

EVIDENCE YOU FOUND:
{evidence}

Respond with JSON:
{{
    "answer": "Direct answer to the question based on evidence",
    "key_findings": ["finding 1", "finding 2", "finding 3"],
    "confidence": 0.0-1.0,
    "gaps": ["What's still missing or unclear"],
    "sources_used": {sources_count}
}}
"""

# ---------------------------------------------------------------------------
# Synthesis (multi-agent)
# ---------------------------------------------------------------------------

SYNTHESIZE_PROMPT = """You are a research synthesizer. Combine findings from multiple research workers.

ORIGINAL QUERY:
{query}
{clarification_section}

WORKER FINDINGS:
{worker_findings}

Your job:
1. Combine all findings into a coherent understanding
2. Resolve any conflicts between workers
3. Identify overall gaps
4. Assess overall confidence

Respond with JSON:
{{
    "combined_answer": "Synthesized answer to the original query",
    "key_themes": ["theme 1", "theme 2"],
    "conflicts": ["Any conflicting information found"],
    "overall_gaps": ["Gaps that remain after all workers"],
    "overall_confidence": 0.0-1.0,
    "needs_more_research": true/false,
    "additional_searches": ["search if needs_more_research is true"]
}}
"""

# ---------------------------------------------------------------------------
# Report writing (multi-agent)
# ---------------------------------------------------------------------------

WRITE_MULTI_PROMPT = """Write a comprehensive research report synthesizing findings from multiple research workers.

QUERY: {query}
{clarification_section}

WORKER FINDINGS:
{worker_summaries}

ALL EVIDENCE:
{evidence}

SOURCES:
{sources}

{structure_instruction}

Rules:
1. Organize by themes/topics, not by worker
2. Every claim needs a citation [1], [2], etc.
3. Note any conflicting information
4. Acknowledge gaps
"""
