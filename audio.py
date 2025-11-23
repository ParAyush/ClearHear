import re
from google import genai


def analyze_audio_file(audio_path: str, api_key: str) -> dict:
    """
    Analyze an audio file using Gemini and return two text outputs:

      - raw_transcript:
          A readable, structured transcript of what was said
          (paragraphs, light headings if helpful, but no summarization).

      - accessible_transcript:
          Detailed, visually clear notes at an approximate second-year university level,
          with headings, timestamps, speaker + tone tags, short quotes, explanations,
          definitions, section summaries, and an overall summary.
    """
    if not api_key:
        raise RuntimeError("No Gemini API key provided to analyze_audio_file().")

    client = genai.Client(api_key=api_key)

    print(f"Uploading audio to Gemini: {audio_path}")
    uploaded_file = client.files.upload(file=audio_path)


    prompt = """
You are an expert academic note-taker assisting a hard-of-hearing student in their second year of university.

You are given an audio recording of a conversation or lecture.

Your job is to produce TWO separate Markdown text blocks in a SINGLE response:

1. A readable raw transcript.
2. Detailed accessible notes.

You MUST return them in EXACTLY this template (do NOT add or remove markers):

<<<RAW_TRANSCRIPT_START>>>
[Put the complete raw transcript here, in Markdown. Use headings and paragraphs, but DO NOT summarize.]
<<<RAW_TRANSCRIPT_END>>>

<<<ACCESSIBLE_NOTES_START>>>
[Put the detailed accessible notes here, in Markdown, using the structure described below.]
<<<ACCESSIBLE_NOTES_END>>>

------------------------
DETAILS FOR EACH PART:
------------------------

RAW TRANSCRIPT
- A faithful written transcript of what was said.
- Use clear paragraphs, punctuation, and sentence boundaries.
- You may add LIGHT structure (e.g., "## Introduction", "## Example") if it helps readability.
- Clean up obvious filler words ("um", "uh", repeated fragments) when they don't add meaning.
- Keep the wording, nuance, and technical vocabulary as close to the original as possible.
- Do NOT summarize or compress the content; this is a transcript, not notes.
- Avoid a giant blob: break into logical paragraphs.

ACCESSIBLE NOTES
- High-quality, detailed lecture notes aimed at a second-year university student.
- Do NOT oversimplify concepts; keep key terminology and nuance.
- The notes MUST be formatted in a neat, visually organized way for accessibility:
    * clear headings
    * short paragraphs
    * bullet points
    * consistent tag formatting
- The notes MUST be DETAILED and THOROUGH enough that the student could study for an exam
  using only these notes.

Use this overall Markdown structure INSIDE <<<ACCESSIBLE_NOTES_START>>> ... <<<ACCESSIBLE_NOTES_END>>>:

# Accessible Notes – [Insert Topic/Title]

## Overview
- 1–3 bullets describing the main themes or goals of the lecture/conversation.

## Main Notes

### [Section Name]
- [T=MM:SS–MM:SS] [Speaker: ROLE] [TONE: tone_word] "SHORT QUOTE OR PHRASE" — Your explanation of what this means and why it matters.
- [T=MM:SS–MM:SS] [Speaker: ROLE] [TONE: tone_word] "Another important line..." — Deeper explanation, context, or example.
- Add as many bullets as needed to capture the full detail of that section.

Requirements for each bullet:
- Include approximate timestamps in [T=MM:SS–MM:SS] format.
- Include the speaker tag: Professor, Student, Guest, TA, Interviewer, Interviewee, etc.
- Include a tone tag when useful:
    calm, friendly, excited, urgent, frustrated, angry, sad, joking, uncertain, neutral.
- Include a short direct quote in quotation marks ("...") that reflects the speaker's wording.
- After the quote, use an em dash (—) and write a clear explanation in complete sentences
  at a second-year university level.
- Capture:
    * main claims or arguments
    * definitions
    * important examples and applications
    * clarifications and distinctions
    * questions or objections raised by students
    * instructions (deadlines, assignments) if present

#### Key Definitions (within a section if relevant)
- When the audio contains definitions of important terms, clearly mark them using:
    - [T=MM:SS–MM:SS] [Speaker: ROLE] [TONE: tone_word] "QUOTE" — **Definition – TERM:** explanation here...

At the end of EACH section, add:

**Section Summary**
- 2–4 short bullets summarizing the most important takeaways of that section, written in your own words (not just quotes).

## Overall Summary
- At the very end of the notes, provide a short summary (3–6 bullets) of the entire lecture,
  as if the student is reviewing before an exam.

VERY IMPORTANT:
- Do NOT output JSON.
- Do NOT output anything before <<<RAW_TRANSCRIPT_START>>>.
- Do NOT output anything after <<<ACCESSIBLE_NOTES_END>>>.
- Do NOT change the marker text.
""".strip()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, uploaded_file],
    )

    full_text = (response.text or "").strip()


    raw_transcript = ""
    accessible_notes = ""

    raw_match = re.search(
        r"<<<RAW_TRANSCRIPT_START>>>\s*(.*?)\s*<<<RAW_TRANSCRIPT_END>>>",
        full_text,
        re.DOTALL,
    )
    notes_match = re.search(
        r"<<<ACCESSIBLE_NOTES_START>>>\s*(.*?)\s*<<<ACCESSIBLE_NOTES_END>>>",
        full_text,
        re.DOTALL,
    )

    if raw_match:
        raw_transcript = raw_match.group(1).strip()
    else:
        raw_transcript = full_text

    if notes_match:
        accessible_notes = notes_match.group(1).strip()
    else:
        accessible_notes = "Accessible notes could not be parsed from the model response.\n\n" + full_text

    return {
        "raw_transcript": raw_transcript,
        "accessible_transcript": accessible_notes,
    }
