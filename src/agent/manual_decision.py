from rapidfuzz import process, fuzz
import uuid
import re
from typing import Dict, Any, Optional


def split_sentences(text: str) -> list[str]:
    """
    Split text based on sentences, handling abbreviations and edge cases.
    """
    if not text or not text.strip():
        return []

    # Common abbreviations (expand as needed)
    abbreviations = {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
        "etc.", "e.g.", "i.e.", "vs.", "st.", "no.", "fig."
    }

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Protect ellipsis
    text = text.replace("...", "<ELLIPSIS>")

    # Protect decimal numbers (e.g., 3.14)
    text = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', text)

    sentences = []
    start = 0

    # Regex for sentence boundary
    pattern = re.compile(r'[.!?]+["\')\]]*\s+')

    for match in pattern.finditer(text):
        end = match.end()
        candidate = text[start:end].strip()

        # Check abbreviation at end
        last_word = candidate.lower().split()[-1]
        if last_word in abbreviations:
            continue

        sentences.append(candidate)
        start = end

    # Add remaining text
    if start < len(text):
        sentences.append(text[start:].strip())

    # Restore protected tokens
    cleaned = []
    for s in sentences:
        s = s.replace("<ELLIPSIS>", "...")
        s = s.replace("<DECIMAL>", ".")
        cleaned.append(s.strip())

    return cleaned

def extract_url_regex(question: str) -> str | None:
    """
    Extract a URL from a string using regex.

    Args:
        question (str): Input text.

    Returns:
        str | None: The extracted URL if found, else None.
    """
    pattern = r'https?://[^\s]+'
    match = re.search(pattern, question)
    return match.group(0) if match else None

def fuzzy_match(question: str, matches: list, threshold: int = 80):
    """
    Fuzzy match a question against a list of strings and return the end index of
    the best match if above threshold.

    Args:
        question (str): The input question.
        matches (list): List of strings to match against.
        threshold (int): Minimum score to consider a match (default 80).

    Returns:
        int | None: The end index of the matched substring in the question if
                    best score >= threshold, else None.
    """
    best_alignment = None
    best_score = 0
    best_match = None

    for match in matches:
        alignment = fuzz.partial_ratio_alignment(match.lower(), question.lower())
        if alignment.score > best_score:
            best_score = alignment.score
            best_alignment = alignment
            best_match = match

    if best_score >= threshold:
        # print("Matched alias:", best_match)
        # print("Start index:", best_alignment.dest_start)
        # print("End index:", best_alignment.dest_end)
        # print("Matched text:", question[best_alignment.dest_start:best_alignment.dest_end])
        return best_alignment.dest_end
    else:
        return None

def extract_url_regex(text: str) -> Optional[str]:
    """
    Extract URL from text using regex.
    """
    url_pattern = r"https?://[^\s]+"
    match = re.search(url_pattern, text)
    return match.group(0) if match else None


def detect_method(text: str) -> str:
    """
    Detect HTTP method from natural language.
    Defaults to GET if nothing matched.
    """
    method_aliases = {
        "GET":    ["get", "fetch", "retrieve", "show", "list", "find", "search", "check"],
        "POST":   ["post", "create", "add", "insert", "submit", "send"],
        "PUT":    ["put", "update", "modify", "edit", "change", "replace"],
        "PATCH":  ["patch", "partial update", "partially update"],
        "DELETE": ["delete", "remove", "destroy", "drop"],
    }

    for method, aliases in method_aliases.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return method

    return "GET"

def normalize_header_name(header: str) -> str:
    """
    Convert header name to canonical HTTP case.
    Example:
        authorization -> Authorization
        content-type -> Content-Type
    """
    return "-".join(word.capitalize() for word in header.split("-"))

def detect_headers(text: str) -> Optional[Dict[str, str]]:
    """
    Detect headers and extract actual values from natural language.
    Supports:
      - 'Authorization': 'Bearer XXX'
      - Authorization: Bearer XXX
      - header Authorization = Bearer XXX
    """

    headers = {}

    # 1️⃣ Detect structured header definitions
    # Matches:
    # 'Authorization': 'Bearer SOME-VALUE'
    # Authorization: Bearer SOME-VALUE
    header_pattern = r"""
        ['"]?([A-Za-z0-9\-]+)['"]?      # header key
        \s*[:=]\s*
        ['"]?([^'"]+)['"]?              # header value
    """

    matches = re.findall(header_pattern, text, re.VERBOSE)

    for key, value in matches:
        # Basic validation: avoid capturing unrelated patterns
        if key.lower() in ["authorization", "content-type", "accept", "x-user-role"]:
            normalized_key = normalize_header_name(key)
            headers[normalized_key] = value.strip()

    # 2️⃣ Fallback intelligent detection if no structured match found
    if not headers:
        if re.search(r"\b(auth|token|bearer|authorization)\b", text):
            bearer_match = re.search(r"bearer\s+([a-zA-Z0-9\-\._]+)", text)
            if bearer_match:
                headers["Authorization"] = f"Bearer {bearer_match.group(1)}"
            else:
                headers["Authorization"] = "Bearer <token>"

        if re.search(r"\bjson\b", text):
            headers["Content-Type"] = "application/json"

        if re.search(r"\bxml\b", text):
            headers["Content-Type"] = "application/xml"

    return headers if headers else None


def detect_payload(text: str, method: str) -> Optional[Dict[str, Any]]:
    """
    Extract payload fields from natural language.
    Only applies to POST, PUT, PATCH.
    """
    if method not in {"POST", "PUT", "PATCH"}:
        return None

    payload = {}

    # Common field aliases
    payload_field_aliases = {
        "service_name": ["service", "plan name", "product"],
        "tier": ["tier", "level", "grade"],
        "name": ["name", "username", "full name"],
        "status": ["status", "state"],
        "role": ["role", "permission", "access level"],
        "service_plan": ["plan", "subscription", "package"],
    }

    # Pattern examples:
    # name is john
    # name=john
    # name: john
    kv_pattern = r"({alias})\s*(?:is|=|:|to|as|with)?\s*([a-zA-Z0-9_\-\.]+)"

    for field, aliases in payload_field_aliases.items():
        for alias in aliases:
            pattern = kv_pattern.format(alias=re.escape(alias))
            match = re.search(pattern, text)
            if match:
                payload[field] = match.group(2)
                break

    return payload if payload else None


def detect_url(text: str) -> Optional[str]:
    """
    Detect URL either from direct URL or known aliases.
    """
    url_aliases = {
        "http://ip-api.com/json/24.48.0.1": ["api location", "ip location"],
    }

    for endpoint, aliases in url_aliases.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return endpoint

    return extract_url_regex(text)


def api_detection(question: str) -> Dict[str, Any]:
    """
    Extract url, method, headers, and payload from a natural language question.
    """

    question_lower = question.lower()

    method = detect_method(question_lower)
    url = detect_url(question_lower)
    headers = detect_headers(question_lower)
    payload = detect_payload(question_lower, method)

    return {
        "url": url,
        "method": method,
        "headers": headers,
        "payload": payload,
    }

def query_detection(question):
    """
    Detect database query components (table, column, value) from a natural language question.

    Uses schema, table aliases, column aliases, and fuzzy matching to identify table,
    column, and value. Returns a dictionary with table and filters.

    Args:
        question (str): The input question.

    Returns:
        dict | None: Dictionary with 'table' and 'filters' keys if table detected,
                     else None. Filters may be empty if column/value not found.
    """
    # 2. Database schema
    database_schema = {
        "dataset_metadata": ["id", "version", "last_updated", "description"],
        "policies": ["policy_id", "title", "category", "description", "role_scope"],
        "policy_rules": ["policy_id", "rule_order", "rule_text"],
        "sla_lookup": ["service_name", "tier", "response_time", "resolution_time", "availability", "support_channels", "escalation_available"],
        "accounts": ["user_id", "name", "role", "status", "service_plan", "last_login"],
        "system_status": ["id", "current_load_percentage", "active_incidents", "system_health", "maintenance_mode", "last_updated"]
    }

    # 3. Table aliases
    table_aliases = {
        "dataset_metadata": ["metadata", "database info"],
        "policies": ["policy"],
        "policy_rules": ["policy rule", "rule details", "rules"],
        "sla_lookup": ["sla", "service level agreement"],
        "accounts": ["account", "user"],
        "system_status": ["status", "system health", "system"]
    }

    # 4. Column aliases — alternative ways to refer to each column
    column_aliases = {
        # dataset_metadata
        "id":                       ["identifier", "user id"],
        "version":                  ["ver"],
        "last_updated":             ["updated", "modified", "last modified", "update date"],
        "description":              ["desc", "details", "info", "about"],

        # policies
        "policy_id":                ["policy number", "policy code", "pol id"],
        "title":                    ["name", "heading", "label", "policy name"],
        "category":                 ["type", "group", "kind", "policy type"],
        "role_scope":               ["role", "scope", "applicable role", "who"],

        # policy_rules
        "rule_order":               ["order", "sequence", "priority", "rule number"],
        "rule_text":                ["rule", "rule content", "rule description", "text"],

        # sla_lookup
        "service_name":             ["service", "plan name", "product", "service type"],
        "tier":                     ["level", "grade", "support tier", "plan tier"],
        "response_time":            ["response", "first response", "initial response", "reply time"],
        "resolution_time":          ["resolution", "fix time", "resolve time", "time to resolve"],
        "availability":             ["uptime", "hours", "available hours", "operating hours"],
        "support_channels":         ["channel", "contact method", "support method", "how to contact"],
        "escalation_available":     ["escalation", "can escalate", "escalate"],

        # accounts
        "user_id":                  ["uid", "id","user id"],
        "name":                     ["user name", "full name", "username", "account name"],
        "role":                     ["user role", "permission", "access level", "position"],
        "status":                   ["account status", "active", "state", "enabled"],
        "service_plan":             ["service plan","plan", "subscription", "package", "tier"],
        "last_login":               ["login", "last seen", "last active", "last access"],

        # system_status
        "current_load_percentage":  ["load", "cpu load", "usage", "load percentage", "current load"],
        "active_incidents":         ["incidents", "issues", "outages", "ongoing incidents"],
        "system_health":            ["health", "system state", "overall health", "health status"],
        "maintenance_mode":         ["maintenance", "in maintenance", "under maintenance"],
    }

    # 5. Extract TABLE
    table = None

    # Check table keyword and extract table name.
    for table_name in table_aliases:
        if table_name.lower() in question.lower(): # Exact match
            table = table_name
            break
        if fuzzy_match(question, [table_name], threshold=80): # Fuzzy match
            table = table_name
            break
    if table == None:
        # Alternative name check
        for table_name, aliases in table_aliases.items():
            if any(alias.lower() in question.lower() for alias in aliases):
                table = table_name
                break
            if fuzzy_match(question, aliases, threshold=80):
                table = table_name
                break

    # 6. Extract COLUMN — check direct name + aliases
    column = None
    col_index = -1
    if table:
        for col in database_schema[table]:
            # Direct match
            if col.lower() in question.lower():
                column = col
                col_index = question.lower().find(col.lower()) + len(column)
                break
            
            column_fuzzy = fuzzy_match(question, [col], threshold=80)

            if column_fuzzy:
                column = col
                col_index = column_fuzzy + 1
                break

            # Alias match
            aliases = column_aliases.get(col, [])
            if any(alias.lower() in question.lower() for alias in aliases):
                column = col
                # Find which alias matched
                for alias in aliases:
                    idx = question.lower().find(alias.lower())
                    if idx != -1:
                        col_index = idx + len(alias)
                        break
                break

            column_fuzzy = fuzzy_match(question, aliases, threshold=80)
            if column_fuzzy:
                column = col
                col_index = column_fuzzy + 1
                break

    # 7. Extract VALUE
    value = None
    
    if column:
        if col_index != -1:
            after_column = question[col_index:].strip()
            for filler in ["is", "=", ":", "for", "equals", "?"]:
                after_column = after_column.replace(filler, "").strip()
            value = after_column.strip() if after_column else None
    

    if table == None:
        return None
    elif column == None or value == None:
        return {"table": table, "filters": {}}
    else:
        return {"table": table, "filters": {column: value}}

def manual_decision(messages):
    """
    Determine which tool to call based on the user's messages.

    Splits the latest user message into sentences, applies API detection and query detection
    to each, and generates tool calls if appropriate.

    Args:
        messages (list): List of message dictionaries with 'role' and 'content' keys.

    Returns:
        dict | None: A dictionary containing role, content, thinking, and tool_calls
                     if any tool calls are generated, else None.
    """
    # Get the latest user question
    question = next((msg["content"] for msg in messages if msg.get("role") == "user"), None)

    # Split the question based on sentences
    question= split_sentences(question)

    # If no question, return None
    if not question:
        return None
    
    tool_calls = []

    # Loop through each sentence and decide on tool call
    for index, value in enumerate(question):
        api_detect = api_detection(value)
        if api_detect['url'] != None:
            tool_dict = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "function": {
                    "index": index,
                    "name": "External_API_Simulation_Tool",
                    "arguments": api_detect
                }
            }
            tool_calls.append(tool_dict)
        else:
            query_detect = query_detection(value)
            if query_detect != None:
                tool_dict = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "index": index,
                        "name": "Structured_Data_Query_Tool",
                        "arguments": query_detect
                    }
                }
                
                tool_calls.append(tool_dict)
    if tool_calls:
        #return {"role": "assistant", "content": "", "thinking": "", "tool_calls": tool_calls}
        return {
            "role": "assistant",
            "content": "",  # Keep minimal; guidance goes in thinking
            "thinking": f"""[Tool Selection: Keyword-Matched]
        🔎 Validation Required:
        1. Are the function called relevant to the question?
        2. Are the function arguments relevant to the question?
        3. If YES → proceed, use tool output to formulate answer
        4. If NO → either: (a) corrected tool calling, or (b) refuse to answer the question because lack of capability""",
            "tool_calls": tool_calls
        }
    else:
        return None

if __name__ == "__main__":
    
    print(f"Query detection test {'-'*25}\n")
    messages = "What account plan for username Alice Tan?"
    print("Question:", messages, "\nResults:", query_detection(messages))
    print(f"\nApi detection test {'-'*25}\n")
    messages = "send get request to http://ip-api.com/json/122.50.6.195 and tell me the location?. What is the SLA for service Premium Support?"
    print("Question:", messages, "\nResults:", api_detection(messages))
    print(f"\nManual decision test {"-"*25}\n")
    messages = [
        {
            "role": "user",
            "content": "SLA for policy id POL-001"
        }
    ]

    result = manual_decision(messages)

    print(
        "Question:", messages,
        "\nResults:", result if result.get("tool_calls") else None
    )

    
