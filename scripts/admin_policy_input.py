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
    publish_submission
)

def main():
    print("=== Admin Policy Input Pipeline (V1) ===")
    title = input("Enter policy title: ").strip()
    print("Enter raw policy text (Press Ctrl-D or Ctrl-Z on Windows to finish):")
    raw_text = sys.stdin.read().strip()
    
    if not title or not raw_text:
        print("Title and raw text are required.")
        return
        
    print("\n[1/4] Creating submission...")
    sub = create_submission(title, raw_text)
    
    print("[2/4] Extracting structured data...")
    extract_submission(sub)
    
    print("[3/4] Validating...")
    validate_submission(sub)
    
    from adaptive_ingestion.admin_input_pipeline import generate_clarification_questions
    
    while sub.status == "needs_clarification":
        print("\nI need more information before this can be published:")
        qs = generate_clarification_questions(sub)
        for q in qs:
            print(f"- {q}")
            
        ans = input("\nProvide clarification (or press Enter to skip): ").strip()
        if ans:
            sub.raw_text += f"\nClarification: {ans}"
            print("[2/4] Re-extracting structured data...")
            extract_submission(sub)
            print("[3/4] Re-validating...")
            validate_submission(sub)
        else:
            break
            
    print("\n--- Extraction Result ---")
    print(f"Status: {sub.status}")
    print(f"Confidence: {sub.confidence:.2f}")
    print(json.dumps(sub.extracted_json, indent=2))
    
    if sub.status == "needs_clarification":
        print("\nPublishing blocked: Policy still needs clarification.")
        return
        
    print("\n--- Preview ---")
    preview = preview_submission(sub, question="What does this policy say?")
    if "error" in preview:
        print(f"Preview Error: {preview['error']}")
    else:
        print(f"Sample Question: {preview['sample_question']}")
        print(f"Simulated Chatbot Answer:\n{preview['simulated_answer']}")
        
    if sub.status == "needs_review":
        print("\nWARNING: Submission flagged as 'needs_review' due to missing fields.")
    
    choice = input("\nPublish this policy? (y/n): ").strip().lower()
    if choice == 'y':
        print("[4/4] Publishing...")
        try:
            policy_id = publish_submission(sub)
            print(f"Successfully published! Policy ID: {policy_id}")
        except Exception as e:
            print(f"Failed to publish: {e}")
    else:
        print("Publish aborted.")

if __name__ == "__main__":
    main()
