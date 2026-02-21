from google import genai
from google.genai import types
import os
import asyncio
import json
from PIL import Image


class BJJTechniqueAnalyzer:
    def __init__(self, api_key, taxonomy_path=None):
        if not api_key:
            raise ValueError("API Key is required for BJJTechniqueAnalyzer")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash-thinking-exp-01-21"
        if taxonomy_path is None:
            taxonomy_path = os.path.join(os.path.dirname(__file__), "bjj_analysis_taxonomy.md")
        self.taxonomy_path = taxonomy_path

    def analyze_sequence(self, frames, frame_indices, previous_context=None):
        if not frames:
            return "No frames."

        try:
            with open(self.taxonomy_path, "r") as f:
                taxonomy_text = f.read()
        except Exception:
            taxonomy_text = "Taxonomy file not found."

        system_instruction = f"""
        {taxonomy_text}

        You are analyzing a chunk of video frames.
        """

        prompt = f"""
        Here is a sequence of {len(frames)} frames from a BJJ match.
        The frames correspond to the following frame numbers: {frame_indices}.

        PREVIOUS CONTEXT: "{previous_context if previous_context else "Start of the match."}"

        INSTRUCTIONS:
        1. **Analyze the Flow**: Use frame numbers to define start/end points.
        2. **Resolve Ambiguity**: Use biomechanics.
        3. **Context Awareness**: Use PREVIOUS CONTEXT.
        4. **Identify the Actor**: For each technique, identify which athlete performs it by their visual appearance (e.g. "athlete in white gi", "athlete in blue gi", "top athlete", "bottom athlete").
        5. **Generate Output**: Return ONLY valid JSON matching this format:
        {{
            "current_context_summary": "string",
            "clips": [
                {{
                    "start_frame": int,
                    "end_frame": int,
                    "category": "STANDUP_GAME|GUARD_PLAY|GUARD_PASSING|POSITIONAL_DOMINANCE|SUBMISSION_OFFENSE|DEFENSE_ESCAPES",
                    "specific_technique": "string",
                    "role": "string describing which athlete performs the technique",
                    "reasoning": "string",
                    "confidence": 0.0-1.0
                }}
            ]
        }}
        """

        try:
            contents = [prompt]
            contents.extend(frames)

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                ),
            )

            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return text.strip()
        except Exception as e:
            return json.dumps({"error": str(e)})


class BJJMultiAgentAnalyzer:
    def __init__(self, api_key, taxonomy_path=None):
        if not api_key:
            raise ValueError("API Key is required for BJJMultiAgentAnalyzer")
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3-flash-preview"
        if taxonomy_path is None:
            taxonomy_path = os.path.join(os.path.dirname(__file__), "bjj_analysis_taxonomy.md")
        self.taxonomy_path = taxonomy_path

    async def run_agent(self, role_name, system_prompt, frames, frame_indices, context, temperature=0.7):
        """Runs a single agent with a specific persona."""
        system_instruction_base = """
        You are a BJJ Analyst. Analyze the video frames provided.
        """

        prompt = f"""
        You are acting as the {role_name}.
        Analyze these frames ({len(frames)} frames).
        Frame Indices: {frame_indices}
        Previous Context: "{context}"

        Output your analysis. Be specific and detailed according to your role.
        For each technique or action identified, specify which athlete performs it
        by their visual appearance (e.g. "athlete in white gi", "athlete in blue gi",
        "top athlete", "bottom athlete").
        Do NOT output JSON. Output a raw analysis paragraph.
        """

        try:
            contents = [prompt]
            contents.extend(frames)

            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction_base + "\n\n" + system_prompt,
                    temperature=temperature,
                ),
            )
            return f"--- {role_name} REPORT ---\n{response.text}\n"
        except Exception as e:
            return f"--- {role_name} FAILED ---\nError: {str(e)}\n"

    async def analyze_sequence_async(self, frames, frame_indices, previous_context=None):
        if not frames:
            return "No frames."

        tasks = []

        # Agent A: The Biomechanist
        tasks.append(
            self.run_agent(
                "Biomechanist",
                "Focus ONLY on the physics. Describe limb positions, joint angles, leverage points, and entanglements. Identify which specific joints are under pressure. Always specify which athlete (by appearance) is performing each action.",
                frames, frame_indices, previous_context, temperature=0.8,
            )
        )

        # Agent B: The Referee
        tasks.append(
            self.run_agent(
                "Referee",
                "Focus on IBJJF/ADCC scoring standards. Has a position been held for 3 seconds? Is a submission 'real' or just a setup? Is the pass complete? Always specify which athlete (by appearance) scores or attacks.",
                frames, frame_indices, previous_context, temperature=0.6,
            )
        )

        # Agent C: The Tactician
        tasks.append(
            self.run_agent(
                "Tactician",
                "Focus on the 'Why'. What is the attacker trying to do? Are they baiting? Describe the strategic flow of the match. Always specify which athlete (by appearance) is driving the action.",
                frames, frame_indices, previous_context, temperature=0.9,
            )
        )

        agent_reports = await asyncio.gather(*tasks)
        full_report_text = "\n".join(agent_reports)

        # The Judge
        judge_prompt = f"""
        You are the Head Judge.
        Read the following reports from your specialized agents.

        {full_report_text}

        YOUR TASK:
        1. Synthesize these views into a single ground truth.
        2. Resolve conflicts.
        3. For each technique, identify which athlete performs it by their visual appearance.
        4. Output the final analysis in the required JSON format:
        {{
            "current_context_summary": "string",
            "clips": [
                {{
                    "start_frame": int,
                    "end_frame": int,
                    "category": "STANDUP_GAME|GUARD_PLAY|GUARD_PASSING|POSITIONAL_DOMINANCE|SUBMISSION_OFFENSE|DEFENSE_ESCAPES",
                    "specific_technique": "string",
                    "role": "string describing which athlete performs the technique",
                    "reasoning": "string",
                    "confidence": 0.0-1.0
                }}
            ]
        }}

        TAXONOMY REFERENCE:
        (See system instruction)
        """

        try:
            try:
                with open(self.taxonomy_path, "r") as f:
                    taxonomy_text = f.read()
            except Exception:
                taxonomy_text = ""

            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=judge_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=taxonomy_text + "\n\nYou are the Synthesizer. Output JSON only.",
                    temperature=0.1,
                ),
            )

            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return text.strip()

        except Exception as e:
            return json.dumps({"error": str(e)})


# Wrapper for synchronous calls
def analyze_sequence_sync(analyzer, frames, frame_indices, previous_context):
    return asyncio.run(
        analyzer.analyze_sequence_async(frames, frame_indices, previous_context)
    )
