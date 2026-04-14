from datetime import datetime
import requests

import requests
import time
from typing import Optional, Dict, Any

# https://beeceptor.com/crud-api/#scrollhere <-- use this for testing

def External_API_Simulation_Tool(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 5,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Safe external API caller for agentic AI.
    Includes timeout, retry, structured response, and fallback behavior.
    """

    method = method.upper()
    attempt = 0

    while attempt < retries:
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=timeout)

            elif method == "POST":
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)

            elif method == "PUT":
                response = requests.put(url, json=payload, headers=headers, timeout=timeout)

            elif method == "DELETE":
                response = requests.delete(url, headers=headers, timeout=timeout)

            else:
                raise ValueError(f"Unsupported method: {method}")

            # Raise error for 4xx / 5xx
            response.raise_for_status()

            # Try JSON parsing
            try:
                data = response.json()
            except ValueError:
                data = response.text

            return {
                "status": "success",
                "status_code": response.status_code,
                "data": data,
            }

        except requests.Timeout:
            print(f"[Retry {attempt+1}] Timeout error")

        except requests.RequestException as e:
            print(f"[Retry {attempt+1}] Request failed: {e}")

        attempt += 1
        time.sleep(1)  # small delay before retry

    # Safe fallback
    return {
        "status": "failed",
        "message": "External API unavailable after retries",
        "data": None,
    }

if __name__ == "__main__":
    url = 'https://ca5370d3fddd2f5a3cf4.free.beeceptor.com/my/api/path'
    headers = {
        'Authorization': 'Bearer SOME-VALUE'
    }
    print(External_API_Simulation_Tool(url, "GET", headers=headers))

