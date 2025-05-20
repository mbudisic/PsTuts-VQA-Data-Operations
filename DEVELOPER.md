# PsTuts-VQA-Data-Operations - Developer Guide 🧑‍💻

## Project Architecture 🏗️

This project is designed for creating and manipulating Visual Question Answering (VQA) datasets for Photoshop tutorials. It uses LangChain, RAGAS, and other libraries to create synthetic question-answer pairs based on video transcripts.

## Core Components 🧩

### 1. Data Loaders 📥

Located in `loader.py`, these classes handle loading and processing video transcript data:

- `VideoTranscriptBulkLoader`: Loads entire transcripts as single documents
- `VideoTranscriptChunkLoader`: Splits transcripts into chunk-level documents
- Helper functions for loading from files and URLs

### 2. Golden Dataset Generation 🌠

Found in `create_golden_dataset.py`, this module:
- Creates knowledge graphs from documents
- Applies transformations to extract relationships
- Generates synthetic question-answer pairs using various query synthesizers
- Uses predefined personas to create contextually relevant questions

## Dependencies 📚

Key dependencies (see `pyproject.toml` for complete list):
- `ragas`: Core framework for RAG evaluation and synthetic data generation
- `langchain_openai`: OpenAI integration for LangChain
- `langchain_core`: Core LangChain components
- `sentence-transformers`: For embedding and similarity operations
- `huggingface_hub`: For model access

## Development Workflow 🔄

### Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install the project with development dependencies
pip install -e ".[dev]"
```

### Code Style 🎨

The project uses:
- Black for formatting (line length: 79)
- Ruff for linting
- MyPy for type checking
- isort for import sorting

You can run checks with:
```bash
black .
ruff check .
mypy .
```

## Adding New Features 🚀

### New Loader Types

To add a new loader type:
1. Extend the `BaseLoader` class from LangChain
2. Implement the `lazy_load()` method to yield `Document` objects

```python
class MyCustomLoader(BaseLoader):
    def __init__(self, data):
        self.data = data
        
    def lazy_load(self) -> Iterator[Document]:
        # Process data and yield Document objects
        for item in self.data:
            yield Document(
                page_content=item["content"],
                metadata={"source": item["source"]}
            )
```

### Custom Query Synthesizers

To create a custom query synthesizer:
1. Extend one of the base synthesizer classes from RAGAS
2. Customize the prompt or logic for generating questions

## Testing 🧪

The project uses pytest for testing. Run tests with:

```bash
pytest
```

## Typical Workflows 🌊

### Creating a New Golden Dataset

```python
# 1. Load documents
from loader import load_VQA_file_from_url
docs, group, _ = load_VQA_file_from_url("https://example.com/data.json")

# 2. Create the golden dataset
from create_golden_dataset import create_golden_dataset
testset = create_golden_dataset(
    docs=docs,
    testset_size=50,
    group_name=group
)

# 3. Save or use the testset
testset.to_pandas().to_csv("my_testset.csv")
```

## Common Pitfalls ⚠️

- Large knowledge graphs can consume significant memory during generation
- OpenAI API calls during dataset generation can incur costs
- Processing large video transcript collections may require batching

## Future Development Ideas 💭

- Add support for multimodal (text + image) data sources
- Implement dataset quality metrics
- Create a web interface for dataset inspection

Happy coding! 🎉 