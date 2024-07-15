Install Ollama

`brew install ollama` or curl -fsSL https://ollama.com/install.sh | sh

Start the server

`ollama serve`

Clone the repo
`git@github.com:prabhakarjuzgar/local_llm.git`

Create venv

`python -m venv .env`

`source .env/bin/activate`

`pip install -r requirements.txt`

Execute/Run fastapi app

`fastapi run main.py`

This will produce at endpoint similar to `http://0.0.0.0:8000`

Append `docs` and paste it in a browser - `http://0.0.0.0:8000/docs`
