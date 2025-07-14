# Architecture Assistant

An AI-powered assistant for architectural design and budgeting, built with Streamlit and LangChain.

## Features

- **Budget Assistant**: Analyze property budgets and get recommendations
- **Design Assistant**: Get architectural design suggestions and technical options
- **Market Analysis**: Get insights about property markets
- **Multi-language Support**: Supports both English and French

## Prerequisites

- Python 3.9+
- [Poetry](https://python-poetry.org/) (recommended) or pip
- [Groq API key](https://console.groq.com/) (for LLM access)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/architect-assistant.git
   cd architect-assistant
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n architect-assistant python=3.9
   conda activate architect-assistant
   ```

3. Install dependencies:
   ```bash
   # Using pip
   pip install -r requirements.txt
   
   # Or using Poetry
   poetry install
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Groq API key to the `.env` file

## Usage

### Running the Application

Use the main script to run different modes:

```bash
# List available modes
python main.py list

# Run the budget assistant (default port: 8501)
python main.py budget

# Run the design assistant (default port: 8502)
python main.py design

# Specify a custom port
python main.py budget --port 8000
```

The application will automatically open in your default web browser. If it doesn't, navigate to `http://localhost:8501` (or your specified port).

### Available Modes

1. **Budget Assistant** (`budget`)
   - Analyze property budgets
   - Get market insights
   - Compare properties

2. **Design Assistant** (`design`)
   - Get architectural style recommendations
   - Explore technical options
   - Generate design concepts

## Project Structure

```
architect-assistant/
├── agents/                # AI agent implementations
│   ├── budget/            # Budget-related agents
│   ├── design_agent.py    # Design agent implementation
│   └── ...
├── streamlit/             # Streamlit application files
│   ├── streamlit_budget_app_fixed.py
│   └── streamlit_design_app.py
├── knowledge_base/        # Knowledge base files
├── .env.example          # Example environment variables
├── main.py               # Main entry point
├── README.md             # This file
└── requirements.txt      # Project dependencies
```

## Configuration

Edit the `.env` file to configure the application:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
LOG_LEVEL=INFO
DEFAULT_LOCALE=en_US
THEME=light
```

## Development

### Adding a New Mode

1. Create a new Streamlit app in the `streamlit/` directory
2. Add a new subcommand in `main.py` to run your app
3. Update the README with documentation for your new mode

### Testing

```bash
# Run tests (when available)
pytest

# Lint the code
flake8 .

# Format the code
black .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [LangChain](https://www.langchain.com/) for the LLM orchestration
- [Groq](https://groq.com/) for the LLM API
