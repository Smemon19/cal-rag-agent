import sys
import json
import os

# Add project root to path if needed so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptive_ingestion.admin_input_pipeline import (
    create_submission,
    extract_submission,
    validate_submission,
    preview_submission,
    publish_submission,
    generate_clarification_questions
)
import argparse

def main():
    parser = argparse.ArgumentParser(description="Admin Policy Input Pipeline (V1)")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM-assisted extraction")
    args = parser.parse_args()

    print("=== Admin Policy Input Pipeline (V1) ===")
    if args.use_llm:
        print("Mode: LLM-assisted extraction")
    else:
        print("Mode: deterministic extraction")
        
    submitted_by = input("Enter your name or email (submitted_by): ").strip()
    title = input("Enter policy title: ").strip()
    print("Enter raw policy text (Type 'DONE' on a new line to finish):")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == 'DONE':
                break
            lines.append(line)
        except EOFError:
            break
            
    raw_text = "\n".join(lines).strip()
    
    if not title or not raw_text:
        print("Title and raw text are required.")
        return
        
    print("\n[1/4] Creating submission...")
    sub = create_submission(title, raw_text, submitted_by=submitted_by)
    
    print("[2/4] Extracting structured data...")
    extract_submission(sub, use_llm=args.use_llm)
    
    print("[3/4] Validating...")
    validate_submission(sub)
    
    while sub.status == "needs_clarification":
        print("\n[!] Needs Clarification:")
        print("I need more information before this can be processed:")
        qs = generate_clarification_questions(sub)
        for q in qs:
            print(f"- {q}")
            
        ans = input("\nProvide clarification (or type 'skip' to skip): ").strip()
        if ans.lower() == 'skip' or not ans:
            break
            
        sub.raw_text += f"\nClarification: {ans}"
        print("[2/4] Re-extracting structured data...")
        extract_submission(sub, use_llm=args.use_llm)
        print("[3/4] Re-validating...")
        validate_submission(sub)
            
    print("\n--- Extraction Result ---")
    
    meta = sub.extracted_json.get("meta", {})
    if meta.get("requested_llm") and meta.get("fallback_used"):
        print("[!] LLM-assisted extraction was requested, but the LLM service was unavailable.")
        print("[!] Falling back to strict deterministic extraction.")
        print("[!] This may cause otherwise valid policies to be flagged as needs_clarification.\n")
        
    print(f"Status: {sub.status}")
    print(f"Confidence: {sub.confidence:.2f}")
    
    item = sub.extracted_json.get("item", {})
    if item:
        print("Extracted Fields:")
        for key, val in item.items():
            if val:
                print(f"  {key.capitalize()}: {val}")
    else:
        print("Extracted Fields: None")
        
    if sub.status == "needs_clarification":
        print("\n[X] Publishing blocked: Policy lacks essential information (needs_clarification).")
        return
        
    if sub.status == "needs_review":
        print("\n[X] Publishing blocked: Policy is missing required fields like source_quote (needs_review).")
        return
        
    if sub.status != "extracted":
        print(f"\n[X] Publishing blocked: Unknown status {sub.status}.")
        return
        
    print("\n--- Preview ---")
    preview = preview_submission(sub, question="What does this policy say?")
    if "error" in preview:
        print(f"Preview Error: {preview['error']}")
    else:
        print(f"Sample Question: {preview['sample_question']}")
        print(f"Simulated Chatbot Answer:\n{preview['simulated_answer']}")
        
    print("\n--- Audit Metadata ---")
    print(f"Submitted By: {sub.submitted_by}")
    print(f"Source Type: {sub.source_type}")
    print(f"Version: {sub.version}")
    
    choice = input("\nPublish this policy? (y/yes to publish, anything else to cancel): ").strip().lower()
    if choice in ('y', 'yes'):
        print("[4/4] Publishing...")
        try:
            policy_id = publish_submission(sub, published_by=sub.submitted_by)
            print(f"Successfully published! Policy ID: {policy_id}")
            print(f"Audit record created with submission_id: {sub.id}")
        except Exception as e:
            print(f"Failed to publish: {e}")
    else:
        print("Publish aborted.")

if __name__ == "__main__":
    main()
