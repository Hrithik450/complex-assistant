import re
import json

def parse_json(raw_response):
    if not raw_response:
        return None
        
    match = re.search(r'\{.*\}', raw_response, re.S)
    if match:
        return json.loads(match.group(0))
    return None