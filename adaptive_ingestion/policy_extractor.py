"""Section-level policy extraction with semantic classification + strict output."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from uuid import uuid4

from adaptive_ingestion.contracts import CandidateJson, FieldMappingValue, StagingPayload, UnmappedConcept
from adaptive_ingestion.schema_dictionary import SchemaDictionary


@dataclass
class ExtractionResult:
    payload: StagingPayload
    confidence: float


class PolicyExtractor:
    """Semantic chunk analysis with strict extraction contract outputs."""

    CHUNK_TYPES = {
        "transcript_metadata",
        "filler",
        "discussion",
        "recommendation",
        "policy_rule",
        "policy_list",
        "procedure",
        "definition",
        "faq",
        "action_item",
    }

    def __init__(self, schema_dictionary: SchemaDictionary, extractor_version: str = "v3-semantic-llm"):
        self.dictionary = schema_dictionary
        self.extractor_version = extractor_version
        self.use_llm = (os.getenv("USE_LLM_SEMANTIC_EXTRACTION", "1").strip() != "0")
        self._llm_model = None
        self._llm_model_name = (os.getenv("SEMANTIC_EXTRACT_MODEL") or os.getenv("CAL_MODEL_NAME") or "gemini-2.0-flash").strip()

    @staticmethod
    def _summarize(text: str, max_len: int = 220) -> str:
        cleaned = PolicyExtractor._normalize_transcript_text(text)
        if not cleaned:
            return ""
        first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
        candidate = first_sentence if len(first_sentence) >= 24 else cleaned
        if len(candidate) <= max_len:
            return candidate
        return candidate[:max_len].rstrip() + "..."

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        stop = {
            "So",
            "All",
            "That",
            "This",
            "And",
            "But",
            "If",
            "Then",
            "OK",
            "Yep",
            "Yeah",
            "Thanks",
            "Thank",
        }
        out: list[str] = []

        # Prefer person names (two or more title-cased tokens)
        for m in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", text or ""):
            if m.split()[0] in stop:
                continue
            if m not in out:
                out.append(m)
            if len(out) >= 6:
                return out

        # Add product / org entities using stricter allowlist-like patterns
        for pat in (r"\bBitwarden\b", r"\bMicrosoft Edge\b", r"\bOneLogin\b", r"\bMFA\b", r"\bCSV\b"):
            for m in re.findall(pat, text or ""):
                if m not in out:
                    out.append(m)
                if len(out) >= 6:
                    return out
        return out

    @staticmethod
    def _normalize_transcript_text(text: str) -> str:
        """Strip speaker/time artifacts and normalize conversational text."""
        if not text:
            return ""
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            # Remove lines like "Lisa Iverson   14:28"
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+\d{1,2}:\d{2}$", line):
                continue
            # Remove single-word conversational fillers
            if line.lower() in {"ok", "okay", "yeah", "yep", "hmm", "bye", "thanks", "thank you"}:
                continue
            lines.append(line)
        cleaned = " ".join(lines)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Trim leading discourse markers
        cleaned = re.sub(r"^(so|well|okay|ok|alright|all right)[,\s]+", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        t = (text or "").strip()
        if not t.startswith("```"):
            return t
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_list_text(text: str) -> str:
        """Normalize list bullets and OCR bullet separators without changing normal prose."""
        if not text:
            return ""
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[•◦▪▫]\s*", "\n- ", cleaned)

        # OCR sometimes turns bullets into " e ". Only treat it as a list separator
        # when there are multiple separators in a short list-like phrase.
        if len(re.findall(r"\s+e\s+", cleaned)) >= 2:
            parts = [p.strip() for p in re.split(r"\s+e\s+", cleaned) if p.strip()]
            item_like = sum(
                1
                for part in parts
                if re.search(r"\b(day|holiday|category|activity|requirement|tool|approval|time)\b", part, re.IGNORECASE)
            )
            if item_like >= 3:
                cleaned = parts[0] + "".join(f"\n- {part}" for part in parts[1:])

        cleaned = re.sub(
            r"(?im)^([^:\n]*(?:holiday|holidays|categor(?:y|ies)|tools?|requirements?|activities|activity codes?)[^:\n]*):\s+([^-*\n].+)$",
            r"\1:\n- \2",
            cleaned,
        )

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _extract_list_items(cleaned_text: str) -> list[str]:
        items: list[str] = []
        for raw in cleaned_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^(?:[-*•]|\d+[\.)]|[A-Za-z][\.)])\s+(.+)$", line)
            if not m:
                continue
            item = re.sub(r"\s+", " ", m.group(1).strip(" ;,"))
            if item and item not in items:
                items.append(item)
        return items

    @staticmethod
    def _list_heading(cleaned_text: str) -> str | None:
        heading_terms = r"(holiday|holidays|categor(?:y|ies)|tools?|requirements?|activities|activity codes?)"
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            if re.search(heading_terms, line, re.IGNORECASE) and ":" in line:
                return line.split(":")[0].strip()
            if re.search(heading_terms, line, re.IGNORECASE) and idx + 1 < len(lines):
                if re.match(r"^(?:[-*•]|\d+[\.)]|[A-Za-z][\.)])\s+", lines[idx + 1]):
                    return line.strip(" :")
        return None

    @staticmethod
    def _topic_from_list_heading(heading: str) -> str:
        low = heading.lower()
        if "holiday" in low:
            return "Holidays"
        if "categor" in low:
            return "Categories"
        if "activit" in low:
            return "Activities"
        if "tool" in low:
            return "Tools"
        if "requirement" in low:
            return "Requirements"
        return heading.title()

    def _extract_complete_list_candidate(
        self,
        *,
        document_id: str,
        section_id: str,
        section_text: str,
    ) -> ExtractionResult | None:
        cleaned = self._normalize_list_text(section_text)
        items = self._extract_list_items(cleaned)
        heading = self._list_heading(cleaned)
        if not heading or len(items) < 3:
            return None

        topic = self._topic_from_list_heading(heading)
        clean_list = heading + ":\n" + "\n".join(f"- {item}" for item in items)
        summary_items = ", ".join(items[:8])
        if len(items) > 8:
            summary_items += f", and {len(items) - 8} more"
        summary = f"{heading} list includes: {summary_items}."

        mapped = {
            "topic": FieldMappingValue(value=topic, confidence=0.9, provenance="deterministic:list_heading"),
            "action_text": FieldMappingValue(value=clean_list, confidence=0.9, provenance="deterministic:complete_list"),
        }

        candidate = CandidateJson(
            candidate_id=f"cand_{uuid4().hex}",
            document_id=document_id,
            section_id=section_id,
            chunk_type="policy_list",
            publishable=True,
            summary=summary,
            topic=topic,
            subtopic=None,
            condition_text=None,
            action_text=clean_list,
            recommendation_text=None,
            entities=self._extract_entities(cleaned),
            source_quote=clean_list,
            reason_if_not_publishable=None,
            rule_text=clean_list,
            confidence=0.9,
            extractor_version=self.extractor_version,
            normalization_notes="Detected and preserved complete list deterministically.",
        )
        payload = StagingPayload(
            candidate_json=candidate,
            mapped_fields_json=mapped,
            unmapped_concepts_json=[],
        )
        return ExtractionResult(payload=payload, confidence=candidate.confidence)

    def _get_llm(self):
        if self._llm_model is not None:
            return self._llm_model
        import google.auth
        from vertexai import init as vertexai_init
        from vertexai.generative_models import GenerativeModel

        creds, detected_project = google.auth.default()
        project_id = (
            os.getenv("VERTEX_PROJECT_ID")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or detected_project
            or "badgers-487618"
        )
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        vertexai_init(project=project_id, location=location, credentials=creds)
        self._llm_model = GenerativeModel(self._llm_model_name)
        return self._llm_model

    def _llm_extract(self, *, normalized_text: str, source_quote: str, document_id: str, section_id: str) -> CandidateJson:
        """LLM-first extraction producing strict, grounded JSON."""
        model = self._get_llm()
        prompt = f"""
You are a semantic policy extraction engine.

Your job:
1) Classify this chunk into one chunk_type from:
transcript_metadata, filler, discussion, recommendation, policy_rule, procedure, definition, faq, action_item
2) Decide if it is publishable.
3) Rewrite messy transcript language into concise normalized statements.
4) Stay fully grounded in the chunk text only; do not invent facts.

Return JSON only, no prose.
Schema:
{{
  "chunk_type": "one of the allowed values",
  "publishable": true/false,
  "summary": "concise normalized statement",
  "topic": "nullable string",
  "subtopic": "nullable string",
  "condition_text": "nullable normalized condition",
  "action_text": "nullable normalized action",
  "recommendation_text": "nullable normalized recommendation",
  "entities": ["strict named entities only"],
  "source_quote": "verbatim supporting quote fragment from the chunk",
  "confidence": 0.0-1.0,
  "reason_if_not_publishable": "required when publishable=false, else null"
}}

Rules:
- Mark publishable=false ONLY for true filler (greetings, timestamps, single-word responses) or transcript_metadata.
- Policy documents contain durable guidance — discussion, recommendation, procedure, and policy_rule chunks from policy documents should generally be publishable=true if they contain any actionable or informational content.
- Publishable procedure/recommendation is allowed even if condition_text is null, as long as action/summary is clear.
- Keep entities strict (people, products, orgs); exclude filler tokens like So, All, That.
- source_quote must be directly copied from chunk text.

CRITICAL — CROSS-REFERENCE RULE:
If this chunk mentions that a DIFFERENT category or policy area applies to a specific situation
(e.g. "for X situation, use Y category instead" or "this time should be charged to Z"),
set the topic to reflect the REFERENCED category (Y or Z), not the section heading this chunk came from.
Example: A note inside a "Professional Development" section that says "if you are leading training,
your time is General Administration" should produce topic="General Administration", not topic="Professional Development".
The topic must reflect what the rule IS ABOUT, not where it physically appears in the document.

Chunk text:
\"\"\"{normalized_text}\"\"\"
"""
        response = model.generate_content(prompt)
        text = self._strip_json_fences(getattr(response, "text", "") or "")
        data = json.loads(text)

        chunk_type = str(data.get("chunk_type") or "").strip()
        if chunk_type not in self.CHUNK_TYPES:
            chunk_type = "discussion"

        publishable = bool(data.get("publishable"))
        confidence = float(data.get("confidence") or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        summary = self._summarize(str(data.get("summary") or normalized_text))
        topic = (data.get("topic") or None)
        subtopic = (data.get("subtopic") or None)
        condition_text = (data.get("condition_text") or None)
        action_text = (data.get("action_text") or None)
        recommendation_text = (data.get("recommendation_text") or None)
        entities_raw = data.get("entities") or []
        entities = [str(x).strip() for x in entities_raw if str(x).strip()]
        entities = self._extract_entities(" ".join(entities) if entities else normalized_text)[:8]
        llm_quote = str(data.get("source_quote") or "").strip()
        if not llm_quote or llm_quote.lower() not in normalized_text.lower():
            llm_quote = source_quote
        reason = data.get("reason_if_not_publishable")
        reason_if_not_publishable = str(reason).strip() if reason not in (None, "") else None
        if not publishable and not reason_if_not_publishable:
            reason_if_not_publishable = f"chunk_type={chunk_type} is non-publishable"

        return CandidateJson(
            candidate_id=f"cand_{uuid4().hex}",
            document_id=document_id,
            section_id=section_id,
            chunk_type=chunk_type,
            publishable=publishable,
            summary=summary,
            topic=topic,
            subtopic=subtopic,
            condition_text=condition_text,
            action_text=action_text,
            recommendation_text=recommendation_text,
            entities=entities,
            source_quote=llm_quote[:600],
            reason_if_not_publishable=reason_if_not_publishable,
            rule_text=normalized_text[:1200],
            confidence=confidence,
            extractor_version=self.extractor_version,
        )

    @staticmethod
    def _classify_chunk(text: str) -> str:
        low = (text or "").lower().strip()
        if not low:
            return "filler"
        if "started transcription" in low or "stopped transcription" in low:
            return "transcript_metadata"
        if re.match(r"^[a-z]+\s+\d{1,2}:\d{2}$", low):
            return "transcript_metadata"
        if re.search(r"\b(thank you|thanks|bye|hmm|okay|ok)\b", low) and len(low.split()) <= 12:
            return "filler"
        if re.search(r"\b(action item|follow up|owner|todo)\b", low):
            return "action_item"
        if re.search(r"\b(recommend|suggest|best practice)\b", low):
            return "recommendation"
        if re.search(r"\b(defined as|definition|means)\b", low):
            return "definition"
        if re.search(r"\b(if|when)\b", low) and re.search(r"\b(should|must|required|need to|cannot|can't)\b", low):
            return "policy_rule"
        if re.search(r"\b(first|next|then|go to|click|set up|enable|open)\b", low):
            return "procedure"
        if "?" in low and re.search(r"\b(how|what|why|where|when)\b", low):
            return "faq"
        return "discussion"

    @staticmethod
    def _extract_condition_action(text: str) -> tuple[str | None, str | None]:
        low = PolicyExtractor._normalize_transcript_text(text)
        cond = None
        action = None
        m_cond = re.search(r"\b(if|when)\b(.+?)(?:,|\\.|;)", low, re.IGNORECASE)
        if m_cond:
            cond = f"{m_cond.group(1).capitalize()}{m_cond.group(2).strip()}".strip()
        m_action = re.search(
            r"\b(should|must|required|need to|cannot|can't|recommend(?:ed)?)\b(.+?)(?:\\.|;|$)",
            low,
            re.IGNORECASE,
        )
        if m_action:
            action = f"{m_action.group(1).capitalize()}{m_action.group(2).strip()}".strip()
        return cond, action

    @staticmethod
    def _topic_subtopic(text: str) -> tuple[str | None, str | None]:
        low = (text or "").lower()
        if "bitwarden" in low or "password" in low or "vault" in low:
            if "mfa" in low or "2fa" in low or "multi-factor" in low:
                return "security", "mfa_setup"
            if "shared collection" in low:
                return "security", "shared_collections"
            return "security", "password_management"
        if "policy" in low:
            return "policy", "general"
        if "recommend" in low:
            return "guidance", "recommendation"
        return None, None

    def extract(self, *, document_id: str, section_id: str, section_text: str) -> ExtractionResult:
        normalized_text = self._normalize_transcript_text(section_text)
        source_quote = normalized_text[:600] if normalized_text else section_text.strip()[:600]
        mapped: dict[str, FieldMappingValue] = {}
        unmapped: list[UnmappedConcept] = []
        low = normalized_text.lower()

        list_result = self._extract_complete_list_candidate(
            document_id=document_id,
            section_id=section_id,
            section_text=section_text,
        )
        if list_result is not None:
            return list_result

        # LLM-first semantic extraction path (grounded in chunk text).
        if self.use_llm:
            try:
                candidate = self._llm_extract(
                    normalized_text=normalized_text or section_text,
                    source_quote=source_quote,
                    document_id=document_id,
                    section_id=section_id,
                )
                if candidate.topic:
                    mapped["topic"] = FieldMappingValue(value=candidate.topic, confidence=candidate.confidence, provenance="llm:topic")
                if candidate.subtopic:
                    mapped["subtopic"] = FieldMappingValue(value=candidate.subtopic, confidence=candidate.confidence, provenance="llm:subtopic")
                if candidate.condition_text:
                    mapped["condition_text"] = FieldMappingValue(value=candidate.condition_text, confidence=candidate.confidence, provenance="llm:condition")
                if candidate.action_text:
                    mapped["action_text"] = FieldMappingValue(value=candidate.action_text, confidence=candidate.confidence, provenance="llm:action")
                if candidate.recommendation_text and "action_text" not in mapped:
                    mapped["action_text"] = FieldMappingValue(value=candidate.recommendation_text, confidence=candidate.confidence, provenance="llm:recommendation")

                payload = StagingPayload(
                    candidate_json=candidate,
                    mapped_fields_json=mapped,
                    unmapped_concepts_json=unmapped,
                )
                return ExtractionResult(payload=payload, confidence=candidate.confidence)
            except Exception as e:
                # Safe fallback to deterministic extraction when LLM path fails.
                print(f"[policy_extractor] LLM extraction failed; falling back. error={e}")

        chunk_type = self._classify_chunk(section_text)
        topic, subtopic = self._topic_subtopic(normalized_text)
        condition_text, action_text = self._extract_condition_action(normalized_text)
        recommendation_text = self._summarize(normalized_text, 280) if chunk_type == "recommendation" else None

        instruction_signal = bool(
            re.search(r"\b(go to|click|open|select|import|export|set up|enable|disable|use)\b", low)
        )

        if topic:
            mapped["topic"] = FieldMappingValue(value=topic, confidence=0.68, provenance="semantic:topic")
        if subtopic:
            mapped["subtopic"] = FieldMappingValue(value=subtopic, confidence=0.64, provenance="semantic:subtopic")
        if condition_text:
            mapped["condition_text"] = FieldMappingValue(value=condition_text, confidence=0.7, provenance="semantic:condition")
        if action_text:
            mapped["action_text"] = FieldMappingValue(value=action_text, confidence=0.72, provenance="semantic:action")
        if recommendation_text:
            mapped["action_text"] = FieldMappingValue(value=recommendation_text, confidence=0.66, provenance="semantic:recommendation")
        elif chunk_type in {"procedure", "recommendation"} and instruction_signal and not action_text:
            action_text = self._summarize(normalized_text, 240)
            mapped["action_text"] = FieldMappingValue(value=action_text, confidence=0.63, provenance="semantic:instructional_action")

        if "policy" in low and "topic" not in mapped:
            mapped["topic"] = FieldMappingValue(value="policy", confidence=0.58, provenance="keyword:policy")
        if "approval" in low:
            mapped["approval_required"] = FieldMappingValue(value=True, confidence=0.7, provenance="keyword:approval")
        if "billable" in low:
            mapped["is_billable"] = FieldMappingValue(value=True, confidence=0.65, provenance="keyword:billable")
        cap_match = re.search(r"\$?\s?(\d{2,7}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per year|annual|yearly)", low)
        if cap_match:
            amt = float(cap_match.group(1).replace(",", ""))
            mapped["amount_threshold"] = FieldMappingValue(value=amt, confidence=0.75, provenance="regex:annual_cap")
            mapped["threshold_unit"] = FieldMappingValue(value="USD", confidence=0.7, provenance="default:currency")

        for raw_label in ("annual_cap", "yearly_cap", "annual_limit", "receipt_required"):
            if raw_label.replace("_", " ") in low:
                normalized = self.dictionary.normalize_name(raw_label)
                if normalized in self.dictionary.canonical_fields():
                    mapped[normalized] = FieldMappingValue(
                        value=True if "required" in raw_label else raw_label,
                        confidence=0.55,
                        provenance=f"alias:{raw_label}",
                    )
                else:
                    unmapped.append(
                        UnmappedConcept(
                            raw_label=raw_label,
                            normalized_label=normalized,
                            observed_values=[raw_label],
                            suggested_classification="new_scalar",
                            frequency_in_batch=1,
                        )
                    )

        tokens = re.findall(r"\b[a-z_]{4,}\b", low)
        for token in sorted(set(tokens)):
            if token in self.dictionary.canonical_fields():
                continue
            if self.dictionary.is_reserved(token):
                continue
            if token.endswith("_required") and not any(u.raw_label == token for u in unmapped):
                unmapped.append(
                    UnmappedConcept(
                        raw_label=token,
                        normalized_label=self.dictionary.normalize_name(token),
                        observed_values=[token],
                        suggested_classification="new_scalar",
                        frequency_in_batch=1,
                    )
                )

        base_conf = {
            "policy_rule": 0.72,
            "procedure": 0.68,
            "recommendation": 0.66,
            "definition": 0.62,
            "faq": 0.58,
            "action_item": 0.64,
            "discussion": 0.45,
            "transcript_metadata": 0.35,
            "filler": 0.2,
        }.get(chunk_type, 0.4)
        confidence = min(0.95, max(0.05, base_conf + 0.04 * len(mapped)))

        semantic_types = {"policy_rule", "procedure", "recommendation", "definition", "faq", "action_item"}
        has_structure = bool(topic or subtopic or condition_text or action_text or recommendation_text)
        publishable = chunk_type in semantic_types and confidence >= 0.50 and has_structure
        # Relaxed gate: strong instructional chunks can publish with slightly weaker structure.
        if not publishable and chunk_type in {"procedure", "recommendation"} and instruction_signal and confidence >= 0.48:
            publishable = True
        reason_if_not_publishable = None
        if not publishable:
            if chunk_type not in semantic_types:
                reason_if_not_publishable = f"chunk_type={chunk_type} is non-publishable"
            elif confidence < 0.50:
                reason_if_not_publishable = f"low confidence ({confidence:.2f})"
            elif not has_structure:
                reason_if_not_publishable = "insufficient structured signal"

        candidate = CandidateJson(
            candidate_id=f"cand_{uuid4().hex}",
            document_id=document_id,
            section_id=section_id,
            chunk_type=chunk_type,
            publishable=publishable,
            summary=self._summarize(section_text),
            topic=topic,
            subtopic=subtopic,
            condition_text=condition_text,
            action_text=action_text,
            recommendation_text=recommendation_text,
            entities=self._extract_entities(normalized_text),
            source_quote=source_quote,
            reason_if_not_publishable=reason_if_not_publishable,
            rule_text=normalized_text[:1200] if normalized_text else section_text[:1200],
            confidence=confidence,
            extractor_version=self.extractor_version,
        )
        payload = StagingPayload(
            candidate_json=candidate,
            mapped_fields_json=mapped,
            unmapped_concepts_json=unmapped,
        )
        return ExtractionResult(payload=payload, confidence=confidence)

