# PsTuts-VQA-Data-Operations 🎨 📊

## About 🌟

This is a toolkit for creating and manipulating Visual Question Answering (VQA) datasets specifically focused on Photoshop tutorials. It helps you transform video transcripts into structured data that can be used for training and evaluating AI systems that answer questions about Photoshop.

## Features ✨

- 📋 Load and process video transcripts from JSON files
- 🔄 Convert transcripts into LangChain Document objects
- 🧠 Create knowledge graphs from document collections
- 🤖 Generate synthetic question-answer pairs for evaluation
- 🧪 Support for creating RAG (Retrieval-Augmented Generation) test datasets

## Getting Started 🚀

### Prerequisites

- Python 3.11 or higher

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PsTuts-VQA-Data-Operations.git
cd PsTuts-VQA-Data-Operations

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install dependencies
pip install -e .
```

## Usage Examples 💡

### Loading Video Transcripts

```python
from loader import VideoTranscriptBulkLoader
import json

# Load from a local JSON file
with open("path/to/your/file.json", "r") as f:
    json_payload = json.load(f)

# Create documents from the transcripts
docs = VideoTranscriptBulkLoader(json_payload=json_payload).load()
```

### Creating a Golden Dataset

```python
from create_golden_dataset import create_golden_dataset

# Generate a test dataset with 10 question-answer pairs
testset = create_golden_dataset(
    docs=your_document_list,
    testset_size=10,
    group_name="photoshop_basics"
)
```

## Contributing 🤝

We welcome contributions! Feel free to open issues or submit pull requests.

## License 📜

This project is licensed under the MIT License - see the LICENSE file for details.
