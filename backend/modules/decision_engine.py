import google.generativeai as genai
import json
import os
import re
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def evaluate_clauses(entities, clauses):
    """Evaluate clauses based on entities and return decision."""
    try:
        logger.info(f"Evaluating clauses with entities: {entities}")
        if not clauses:
            logger.warning("No clauses provided for evaluation")
            return {"decision": "Rejected", "amount": 0, "justification": "No relevant clauses found", "clauses": []}
        
        # Prepare input
        procedure = entities.get("procedure", "").lower()
        policy_duration = entities.get("policy_duration", "")
        pre_approval = entities.get("pre_approval", False)
        
        # Extract duration in months
        duration_months = 0
        if policy_duration:
            match = re.search(r"(\d+)\s*(month|year)", policy_duration, re.IGNORECASE)
            if match:
                num, unit = match.groups()
                duration_months = int(num) if unit.lower() == "month" else int(num) * 12
        
        # Define IRDAI rules
        waiting_periods = {
            "appendectomy": 1,       # 30 days (1 month)
            "knee surgery": 36,      # 36 months
            "joint replacement surgery": 36,
            "surgery": 1,            # General surgery: 30 days
            "operation": 1,
        }
        
        # Check clauses
        relevant_clauses = []
        for clause in clauses:
            clause_lower = clause.lower()
            if any(term in clause_lower for term in ["hospitalization", "treatment", "surgery", "insured", "waiting period"]):
                relevant_clauses.append(clause)
        
        if not relevant_clauses:
            logger.warning("No relevant clauses found")
            return {"decision": "Rejected", "amount": 0, "justification": "No relevant clauses found", "clauses": []}
        
        # Validate waiting period
        decision = {"decision": "Rejected", "amount": 0, "justification": "", "clauses": relevant_clauses}
        if procedure in waiting_periods:
            required_months = waiting_periods[procedure]
            if duration_months >= required_months:
                decision["decision"] = "Approved"
                decision["amount"] = 50000 if procedure == "appendectomy" else 0
                decision["justification"] = f"Policy duration ({policy_duration}) meets {required_months} month waiting period for {procedure}."
                if pre_approval:
                    decision["justification"] += " Pre-approval obtained."
            else:
                decision["justification"] = f"{procedure} requires {required_months} month waiting period."
        else:
            decision["justification"] = f"No waiting period defined for {procedure}."
        
        # Query Gemini for confirmation
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Given:
        - Procedure: {procedure}
        - Policy duration: {policy_duration}
        - Pre-approval: {pre_approval}
        - Clauses: {relevant_clauses}
        - IRDAI waiting periods: {waiting_periods}
        Confirm if the claim is approved or rejected. If approved, suggest an amount (e.g., 50000 for appendectomy). Provide a justification. Return JSON:
        ```json
        {{
            "decision": "Approved or Rejected",
            "amount": number,
            "justification": "string",
            "clauses": [list of clauses]
        }}
        ```
        """
        try:
            response = model.generate_content(prompt)
            gemini_decision = json.loads(response.text.strip("```json\n```"))
            # Merge Gemini decision with local validation
            if gemini_decision["decision"] == decision["decision"]:
                decision["justification"] = gemini_decision["justification"] or decision["justification"]
                decision["amount"] = gemini_decision["amount"] if gemini_decision["decision"] == "Approved" else 0
            else:
                logger.warning(f"Gemini decision {gemini_decision['decision']} conflicts with local decision {decision['decision']}")
        except Exception as e:
            logger.error(f"Gemini error: {str(e)}", exc_info=True)
        
        logger.info(f"Decision: {decision}")
        return decision
    except Exception as e:
        logger.error(f"Decision engine error: {str(e)}", exc_info=True)
        return {"decision": "Rejected", "amount": 0, "justification": f"Error evaluating clauses: {str(e)}", "clauses": []}