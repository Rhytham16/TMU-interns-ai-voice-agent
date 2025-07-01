
---

## 🟩 3. `load_documents.md`

```markdown
# load_documents.py – Document Loader

This script is responsible for loading local documents and converting them into vector embeddings using LangChain and ChromaDB.

### Purpose
To populate a vector store with indexed documents so the voice agent can retrieve answers based on user queries.

### Key Actions

- Load `.pdf`, `.txt`, or `.docx` files from the `data/` directory.
- Split documents into chunks.
- Create vector embeddings using OpenAI/other models.
- Store them in ChromaDB for fast similarity search.

You can run this file once to prepare your data:

```bash
python load_documents.py
 
