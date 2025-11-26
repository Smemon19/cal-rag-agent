"""
Modules to break down the monolithic retrieve() function.
"""
import re
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import itertools
from dataclasses import dataclass

from utils import build_section_search_terms, format_results_as_context
from utils_tables import parse_markdown_table, pick_underlayment
from rules_loader import find_rules_by_section, find_rules_by_keywords
from utils_vectorstore import VectorStore


class QueryPreprocessor:
    """Handles query normalization and token extraction."""
    
    @staticmethod
    def extract_section_tokens(query: str) -> List[str]:
        """Extract potential section numbers like '1507.2'."""
        return re.findall(r"\b\d+(?:\.\d+)+\b", query or "")

    @staticmethod
    def build_search_terms(query: str) -> List[str]:
        """Expand query into search terms using existing utility."""
        return build_section_search_terms(query)


class SpecializedLookup:
    """Handles domain-specific lookups (tables, rules)."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.known_materials = self._load_materials(config_path)

    def _load_materials(self, config_path: Optional[Path]) -> List[str]:
        try:
            if not config_path:
                # Try standard location
                config_path = Path(__file__).resolve().parent / "config" / "materials.json"
            
            if config_path.exists():
                with open(config_path, "r") as f:
                    return json.load(f).get("special_materials", [])
        except Exception as e:
            print(f"[config] Failed to load materials.json: {e}")
        
        # Default fallback
        return [
            "slate shingles", "slate shingle", "slate shingle roof",
            "asphalt shingles", "asphalt shingle", "asphalt shingle roof",
            "vehicle barrier", "vehicle barriers"
        ]

    def lookup_tables(self, query: str, vector_store: VectorStore, n_results: int) -> str:
        """Check for specific material/wind conditions and perform table lookup."""
        ql = (query or "").lower()
        material_hit = next((m for m in self.known_materials if m in ql), None)
        mph_match = re.search(r"(\d{2,3})\s*mph", ql)
        wind_mph_val = int(mph_match.group(1)) if mph_match else None

        if not (material_hit and wind_mph_val is not None):
            return ""

        # Scan candidates
        substrings = [material_hit, "table", "underlayment", "1507"]
        # Use keyword search on vector store
        scan = vector_store.keyword_search(substrings, max_results=max(15, n_results))
        docs = scan.get("documents", [[]])[0]
        metas = scan.get("metadatas", [[]])[0]

        for doc, meta in zip(docs, metas):
            for df in parse_markdown_table(doc or ""):
                val = pick_underlayment(df, material_hit, wind_mph_val)
                if val:
                    citation = self._format_citation(doc, meta)
                    return f"TABLE LOOKUP RESULT: Underlayment for '{material_hit}' at wind speed {wind_mph_val} mph: {val} ({citation})\n\n"
        return ""

    def _format_citation(self, doc: str, meta: Dict) -> str:
        # Helper to format citation from meta
        table_num = None
        m = re.search(r"\btable\s+([0-9]+(?:\.[0-9]+)*)", (doc or ""), flags=re.IGNORECASE)
        if m:
            table_num = m.group(1)
        else:
            sp = str((meta or {}).get("section_path") or (meta or {}).get("headers") or "")
            m2 = re.search(r"([0-9]+(?:\.[0-9]+)+)", sp)
            if m2:
                table_num = m2.group(1)
        
        title = str((meta or {}).get("title") or "")
        src = str((meta or {}).get("source_url") or "")
        code_short = ""
        if "international building code" in (title + src).lower(): code_short = "IBC"
        elif "international existing" in (title + src).lower(): code_short = "IEBC"
        elif "international residential" in (title + src).lower(): code_short = "IRC"
        
        if code_short and table_num: return f"{code_short} Table {table_num}"
        if table_num: return f"Table {table_num}"
        return code_short or "Table lookup"

    def lookup_rules(self, section_tokens: List[str], query: str) -> str:
        """Inject rule snippets based on section numbers or keywords."""
        rule_hits = []
        # Try exact section matches
        for tok in section_tokens:
            rh = find_rules_by_section(tok)
            if rh:
                rule_hits = rh
                break
            # Prefix fallback
            parts = tok.split('.')
            for i in range(len(parts)-1, 0, -1):
                rh = find_rules_by_section('.'.join(parts[:i]))
                if rh:
                    rule_hits = rh
                    break
            if rule_hits: break
        
        # Fallback to keyword search
        if not rule_hits:
            kw_tokens = re.findall(r"[a-zA-Z]+", query or "")
            candidates = find_rules_by_keywords(kw_tokens)
            if candidates:
                rule_hits = [candidates[0][0]]

        if not rule_hits:
            return ""

        # Format rule
        r = rule_hits[0]
        items = r.get("items", [])
        header = f"RULE SNIPPET ({r.get('version')} §{r.get('sec')} – {r.get('title')}):\n"
        lines = []
        for it in items:
            note = f" ({it.get('note')})" if it.get('note') else ""
            lines.append(f"- {it.get('label')}: {it.get('value')}{note}")
        return header + "\n".join(lines) + "\n\n"


class RetrievalOrchestrator:
    """Orchestrates the retrieval process using helper components."""
    
    def __init__(self, vector_store: VectorStore, embedding_function):
        self.vector_store = vector_store
        self.embed_fn = embedding_function
        self.preprocessor = QueryPreprocessor()
        self.lookup = SpecializedLookup()

    def retrieve(self, query: str, n_results: int = 5, filters: Optional[Dict] = None) -> str:
        # 1. Preprocess
        section_tokens = self.preprocessor.extract_section_tokens(query)
        
        # 2. Specialized Lookups
        table_ctx = self.lookup.lookup_tables(query, self.vector_store, n_results)
        rule_ctx = self.lookup.lookup_rules(section_tokens, query)
        
        # 3. Vector/Hybrid Search
        query_embedding = self.embed_fn([query])[0]
        
        # Standard vector search
        vec_res = self.vector_store.vector_search(query_embedding, n_results=n_results, filters=filters)
        
        # Keyword fallback
        kw_res = self.vector_store.keyword_search(
            self.preprocessor.build_search_terms(query), 
            max_results=max(3, n_results)
        )
        
        # 4. Merge & Format
        merged = self._merge_results(vec_res, kw_res, n_results)
        
        # 5. Build Context
        context = format_results_as_context(merged)
        
        # Prepend specialized context
        full_context = ""
        if rule_ctx: full_context += rule_ctx
        if table_ctx: full_context += table_ctx
        full_context += context
        
        return full_context

    def _merge_results(self, vec_res: Dict, kw_res: Dict, n: int) -> Dict:
        """Merge vector and keyword results, deduplicating by ID."""
        # Start with vector results
        ids = vec_res["ids"][0]
        docs = vec_res["documents"][0]
        metas = vec_res["metadatas"][0]
        dists = vec_res["distances"][0]
        
        seen = set(ids)
        
        # Append unique keyword results
        kw_ids = kw_res["ids"][0]
        kw_docs = kw_res["documents"][0]
        kw_metas = kw_res["metadatas"][0]
        # Keyword results might not have distances, default to 0.0
        kw_dists = kw_res.get("distances", [[0.0] * len(kw_ids)])[0]
        
        for i, pid in enumerate(kw_ids):
            if pid not in seen:
                ids.append(pid)
                docs.append(kw_docs[i])
                metas.append(kw_metas[i])
                dists.append(kw_dists[i])
                seen.add(pid)
        
        # Cap at reasonable limit (e.g. 2x requested) before final trim
        limit = max(n * 2, 20)
        return {
            "ids": [ids[:limit]],
            "documents": [docs[:limit]],
            "metadatas": [metas[:limit]],
            "distances": [dists[:limit]]
        }

