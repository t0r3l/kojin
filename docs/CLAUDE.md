# Using your Claude API key

Place your Claude API key in a local environment file or export it in your shell. This file is ignored by git when named `.env`.

Options:

- Add to a `.env` file (recommended for local development):

```
# create a file named .env (copy from .env.example)
CLAUDE_API_KEY=sk-...your_key_here...
CLAUDE_API_URL=https://api.anthropic.com
```

- Or export the key in your shell session:

```
export CLAUDE_API_KEY="sk-...your_key_here..."
export CLAUDE_API_URL="https://api.anthropic.com"
```

Using the key in Python:

```
import os
import requests

api_key = os.getenv('CLAUDE_API_KEY')
base = os.getenv('CLAUDE_API_URL', 'https://api.anthropic.com')
headers = { 'x-api-key': api_key }
resp = requests.post(f"{base}/v1/complete", json={"prompt":"Hello"}, headers=headers)
print(resp.status_code, resp.text)
```

Using curl:

```
curl -s -X POST "$CLAUDE_API_URL/v1/complete" \
  -H "x-api-key: $CLAUDE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a short greeting."}'
```

Notes:

- Do not commit your real `.env` file or API keys to version control.
- Use the `.env.example` as a template for new environments.
- Consider using a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.) for production.

Using a plain `.claude_key` file
--------------------------------

You can store the API key in a file named `.claude_key` containing only the key. This file is already ignored by the repository's `.gitignore`.

Example (create at repository root):

```
echo "sk-...your_key_here..." > .claude_key
chmod 600 .claude_key
```

Helper module
-------------

The project includes a small helper `claude_config.py` that loads the key from `CLAUDE_API_KEY`, a provided file path, `.claude_key`, or a `.env` (via `python-dotenv` if available). Example usage:

```
from claude_config import make_request_json, api_base_url
import requests

body = {"model":"claude-instant-code","prompt":"Write a Python function that reverses a string.","max_tokens_to_sample":300}
resp = make_request_json(requests, "/v1/complete", body)
print(resp.status_code)
print(resp.text)
```

If you prefer explicit control, you can still read the key yourself via `open('.claude_key').read().strip()` or `os.getenv('CLAUDE_API_KEY')`.

