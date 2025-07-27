# Semantic Video Search Engine

This project is an end-to-end pipeline and application that transforms a raw video file into a fully searchable asset. It uses a suite of AI models to understand the video's content, allowing you to search for specific moments using natural language. You can search by describing **what was said** (dialogue) or **what was shown** (visuals).



## Troubleshooting

### Common Issues

1. **Port conflicts**: If you see "Address already in use" errors:
   ```bash
   # Check what's using the port (e.g., 5050)
   lsof -nP -iTCP:5050 | grep LISTEN
   
   # Kill the process if needed
   kill -9 <process_id>
   ```

2. **FFmpeg not found**: Make sure FFmpeg is installed and in your PATH:
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   ```

3. **Model download issues**: The first run downloads large AI models. Ensure you have:
   - Stable internet connection
   - Sufficient disk space (several GB)
   - Patience for the initial setup

### Database Inspection

To inspect your ChromaDB database:
```bash
python inspect_db.py
```

---

## Features

- **Automated Processing Pipeline:** Ingests a raw video and automatically performs all processing steps.
- **Multi-Modal Analysis:** Extracts information from three different modalities:
    - **Audio:** Speaker diarization (who spoke when) and non-speech event detection (music, applause).
    - **Visuals:** Shot detection and AI-powered visual captioning for every scene.
    - **Text:** High-accuracy transcription of all spoken dialogue.
- **Intelligent Segmentation:** Uses a "Boundary Scoring" algorithm to group individual shots into coherent, logical narrative segments.
- **LLM-Powered Enrichment:** Leverages a Large Language Model (like GPT or Gemini) to generate a concise title, summary, and keywords for every segment.
- **Hybrid Semantic Search:** Creates separate vector embeddings for textual and visual data, allowing for powerful and precise hybrid search.
- **Interactive UI:** A simple web interface to search videos and instantly jump to the relevant timestamp.

---

## System Architecture

The project is divided into two main parts: an offline **Ingestion Pipeline** that processes videos and a real-time **Search Application** that serves user queries.

```
+----------------+      +--------------------------+      +----------------------+
|   Raw Video    |----->|  Phase 1: Extraction     |----->|  Processed Files     |
+----------------+      |  - Transcription (WhisperX) |      |  - transcript.json   |
                        |  - Shots (TransNetV2)       |      |  - shots.json        |
                        |  - Visuals (BLIP)           |      |  - visual_details.json|
                        |  - Audio Events (AST)       |      |  - audio_events.json |
                        +--------------------------+      +----------------------+
                                     |
                                     v
+----------------------+      +--------------------------+      +---------------------+
| Speaker ID Tool (UI) |----->|  Phase 2: Segmentation   |----->|  final_segments.json|
+----------------------+      +--------------------------+      +---------------------+
                                     |
                                     v
+----------------------+      +--------------------------+      +-------------------------+
| enriched_segments.json|<----|  Phase 3: LLM Enrichment |      |  Phase 4: Indexing      |
+----------------------+      +--------------------------+      |  - Create Embeddings    |
                                                                |  - Store in ChromaDB    |
                                                                +-------------------------+
                                                                          |
+----------------------+      +--------------------------+                v
| Search UI (Streamlit)|<---->|   Search API (FastAPI)   |<------> [ Vector Database ]
+----------------------+      +--------------------------+          [  (ChromaDB)   ]
```

---

## Tech Stack

- **Backend API:** FastAPI, Uvicorn
- **Frontend UI:** Streamlit
- **Vector Database:** ChromaDB
- **AI/ML Models:**
    - **Transcription:** `WhisperX`
    - **Shot Detection:** `TransNetV2`
    - **Visual Captioning:** `Salesforce/blip-image-captioning-base`
    - **Audio Events:** `MIT/ast-finetuned-audioset-10-10-0.4593`
    - **Embeddings:** `all-MiniLM-L6-v2` (or other sentence-transformers)
    - **Enrichment:** Google Gemini API (primary) or OLLAMA models
- **Core Libraries:** PyTorch, OpenCV, Pandas, FFmpeg, PyYAML, colorlog
- **Web Framework:** FastAPI (with Uvicorn), Streamlit
- **Additional:** google-generativeai, ffmpeg-python

---

## Setup and Installation

Follow these steps to set up the project environment.

### 1. Prerequisites

- **Python 3.10+**
- **FFmpeg:** Must be installed and accessible from your command line.
    - On macOS: `brew install ffmpeg`
    - On Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
- **Docker and Docker Compose:** For running the ChromaDB vector database.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/video-search-engine.git
cd video-search-engine
```

### 3. Set Up Project Structure

Create the necessary directories:

```bash
mkdir -p data/videos
mkdir -p ingestion_pipeline/steps
mkdir -p app/ui
mkdir -p api
```

### 4. Set Up Python Environment

Create and activate a virtual environment (Python 3.12 recommended):

```bash
# Create the virtual environment
python3.12 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
.\venv\Scripts\activate
```

Install the core dependencies:

```bash
# Install PyTorch and related packages
pip install torch torchvision torchaudio

# Install WhisperX and its dependencies
pip install git+https://github.com/m-bain/whisperx.git

# Install TransNetV2
pip install transnetv2-pytorch

# Install other required packages
pip install pandas scikit-learn sentence-transformers opencv-python
pip install ffmpeg-python colorlog PyYAML
pip install transformers opencv-python-headless Pillow librosa timm
pip install streamlit fastapi uvicorn chromadb
pip install google-generativeai
```

Or install from requirements.txt if available:

```bash
pip install -r requirements.txt
```

### 5. Start the Vector Database

Run the ChromaDB Docker container:

```bash
docker run -p 8000:8000 chromadb/chroma
```

### 6. Configure Environment Variables

Create a `config.yaml` file by copying the example, or set environment variables:

```bash
# For Gemini API (recommended based on your setup)
export GEMINI_API_KEY=your_gemini_api_key_here

# Alternative: create config.yaml with your API keys
cp config.example.yaml config.yaml
```

Edit `config.yaml` and fill in your API keys:
- `hf_token`: Your Hugging Face token (for speaker diarization)
- `gemini_api_key`: Your Google Gemini API key (for LLM enrichment)
- `openai_api_key`: Your OpenAI API key (alternative to Gemini)

---

## How to Use

The workflow involves three main stages: processing a video, starting the servers, and using the search app.

### Stage 1: Process a Video

Run the main ingestion pipeline script as a module, pointing it to your video file:

```bash
python -m ingestion_pipeline.run_pipeline --video data/videos/your_video.mp4
```

**Note:** The first time you run this, it will download several large AI models, which may take some time.

### Stage 2: Run the Human-in-the-Loop Speaker ID Tool

After the pipeline finishes Step 1, it will pause. You need to run the speaker identification tool to map generic speaker labels (e.g., `SPEAKER_00`) to real names.

```bash
streamlit run app/ui/speaker_id_tool.py
```

Use the web interface to assign names. Once the `speaker_map.json` is saved, the main pipeline will automatically continue.

### Stage 3: Start the Servers and Search

Once the full pipeline is complete for at least one video, you can start the application servers. **Run each command in a separate terminal window.**

1.  **Start the Backend API:**
    ```bash
    uvicorn api.main:app --reload --port 8001
    ```

2.  **Start the Frontend UI:**
    ```bash
    streamlit run app/ui/search_app.py
    ```

A browser tab should open with the search application. Select your video from the sidebar, type a query, and start searching!

---

## Configuration

The entire pipeline is configurable via the `config.yaml` file. You can change:
- The AI models used for each step (`transcription`, `embedding`, `llm`, etc.).
- The ports for the API server.
- The connection details for the database.
- Default filenames and processing parameters.
