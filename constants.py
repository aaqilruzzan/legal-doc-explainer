"""Constants and configuration values."""

# Clause topics for analysis
CLAUSE_TOPICS = {
    "termination": "Contract Termination",
    "financial": "Payment Terms and Financial Details",
    "liability": "Limitation of Liability and Indemnification",
    "renewal": "Auto-Renewal Clause",
    "service": "Service Level Agreement (SLA) or Scope of Work"
}

# Prompt templates
SUMMARY_PROMPT_TEMPLATE = """
Analyze the following legal document and provide four things in a single JSON object:
1. A summary, generated according to the specific instructions provided.
2. A structured extraction of key data points.
3. A structured extraction of important contract terms.
4. A glossary of important legal terms found in the document.

Return ONLY a valid JSON object with four top-level keys: "summary", "key_data_points", "important_contract_terms", and "legal_terms_glossary". Do not add any text before or after the JSON.

INSTRUCTIONS FOR THE "summary" VALUE:
Provide a clear, easy-to-understand summary of the document. The summary MUST:
- Use plain, simple language and avoid legal jargon.
- Include clear headings for different sections and number them (e.g., "1. Key Responsibilities"). Do not number sub-headings or points.

Example for the "summary" value:
"summary": "1. What This Agreement Is About\\nThis is a service agreement where ABC Company agrees to provide marketing services to XYZ Corporation for a period of one year.\\n\\n2. Key Responsibilities\\nAs the Client, you are responsible for providing necessary materials and feedback in a timely manner. Payments are due on the first of each month."

INSTRUCTIONS FOR THE "key_data_points" VALUE:
This should be an object with the following structure. Be as specific as possible. If a value is not found, use an empty string "" or an empty list [].
{{
  "parties_involved": [ {{ "name": "Name", "role": "Role (use simple role names like 'Provider', 'Customer', 'Contractor' without additional descriptive text in parentheses)" }} ],
  "contract_period": {{
    "start_date": "Extract the specific start date in YYYY-MM-DD format or leave blank if not found.",
    "end_date": "Extract the specific end date in YYYY-MM-DD format or leave blank if not found.",
    "term_description": "Provide a specific description, e.g., 'A 2-year initial term with an option for one 12-month renewal.'"
  }},
  "financial_terms": [
    "Extract specific amounts, fees, and percentages. e.g., '$5,000 monthly subscription fee', '1.5% late fee on overdue payments'"
  ],
  "key_deadlines": [ "e.g., 30-day termination notice" ]
}}

INSTRUCTIONS FOR THE "important_contract_terms" VALUE:
This should be an object containing key-value pairs for high-level contract terms. If a term is not found, use an empty string "".
Example:
{{
  "Service Scope": "Consulting services as defined in Exhibit A",
  "Confidentiality": "Standard NDA provisions apply",
  "Governing Law": "Delaware State Law",
  "Intellectual Property": "Work product ownership defined"
}}

INSTRUCTIONS FOR THE "legal_terms_glossary" VALUE:
This should be an object containing key-value pairs. The key should be 5 complex legal terms found in the document, and the value should be its simple definition.
Example:
{{
    "Force Majeure": "Unforeseeable circumstances that prevent a party from fulfilling a contract.",
    "Indemnification": "Protection against financial loss, typically through compensation."
}}

Document Text to Analyze:
---
{document_text}
---
"""

QA_PROMPT_TEMPLATE = """
You are a helpful assistant analyzing a document. Use the following pieces of context to answer the question at the end.
Your answer should be based ONLY on the provided context.

If the answer is found in the context, provide a clear and concise answer.
If the answer is not found in the context, explicitly state "The provided document does not contain specific information on this topic."

Context:
{context}

Question: {question}

Answer:
"""

CLAUSE_ANALYSIS_TEMPLATE = """You are an expert contract analysis AI.
Analyze the context below related to the topic: {topic_query}.
Based ONLY on the context, generate a structured analysis.

{format_instructions}

CONTEXT:
---
{context_text}
---
"""