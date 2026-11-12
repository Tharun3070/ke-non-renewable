# AI-Powered Renewable Energy Knowledge Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** web application built with **Python**, **Streamlit**, **Google Gemini**, **Sentence Transformers**, **ChromaDB**, and **PyPDF**. Users can upload renewable energy documents (PDF and TXT), index semantic embeddings locally in ChromaDB, ask natural-language questions, and receive strictly grounded AI responses with exact filename, page number, and snippet citations.

---

## 1. Project Structure

```text
renewable-energy-assistant/
│
├── app.py                              # Main Streamlit dashboard (5 navigation sections)
├── requirements.txt                    # Python dependencies
├── .env                                # Environment variables (API key & RAG config)
├── .env.example                        # Template environment file
├── README.md                           # Project documentation
│
├── modules/
│   ├── __init__.py
│   ├── document_processor.py           # PDF/TXT extraction, cleaning, chunking, categorization
│   ├── embeddings.py                   # Sentence Transformers wrapper (all-MiniLM-L6-v2)
│   ├── vector_database.py              # Persistent ChromaDB client, duplicate check, CRUD, search
│   └── rag_pipeline.py                 # Query embedding, retrieval, Gemini grounded generation
│
├── data/
│   ├── generate_sample_pdfs.py         # Script to generate 12 multi-page sample PDFs
│   └── documents/                      # Uploaded & sample PDF/TXT dataset
│       ├── 01_solar_and_wind_energy_systems.pdf
│       ├── 02_hydro_geothermal_biomass_energy.pdf
│       ├── 03_green_hydrogen_and_energy_storage.pdf
│       ├── 04_government_renewable_energy_schemes.pdf
│       ├── 05_solar_photovoltaics_and_csp_engineering.pdf
│       ├── 06_offshore_wind_and_aerodynamics.pdf
│       ├── 07_hydropower_and_pumped_storage_engineering.pdf
│       ├── 08_geothermal_egs_and_district_heating.pdf
│       ├── 09_biomass_biofuels_and_waste_to_energy.pdf
│       ├── 10_green_hydrogen_ammonia_and_transport.pdf
│       ├── 11_grid_scale_battery_and_long_duration_storage.pdf
│       └── 12_global_renewable_policies_and_grid_integration.pdf
│
└── chroma_db/                          # Persistent ChromaDB vector store
```

---

## 2. RAG Architecture Workflow

```text
PDF / TXT Upload
       ↓
Page-by-Page Text Extraction (PyPDF)
       ↓
Text Cleaning & De-hyphenation
       ↓
Overlapping Chunking + Metadata (Filename, Page Number, Category, SHA-256 Hash)
       ↓
Dense Embedding Generation (Sentence Transformers: all-MiniLM-L6-v2)
       ↓
Persistent Vector Storage (ChromaDB - Cosine Space)
       ↓
User Natural-Language Question
       ↓
Question Embedding
       ↓
Top-K Cosine Similarity Search + Threshold Filtering
       ↓
Grounded Context Prompt (Strict Anti-Hallucination Instructions)
       ↓
Google Gemini API (gemini-3.1-flash-lite)
       ↓
Verified Answer + Expandable Source References (Filename, Page, Snippet)
```

---

## 3. Supported Renewable Energy Knowledge Domains

1. **Solar Energy** — PV modules (monocrystalline, polycrystalline, perovskite), MPPT, CSP, trackers
2. **Wind Energy** — Onshore & offshore turbines, Betz limit, power curve, capacity factors
3. **Hydroelectric Energy** — Impoundment, run-of-river, pumped storage, Pelton/Francis/Kaplan turbines
4. **Geothermal Energy** — Dry steam, flash steam, binary cycle (ORC), Enhanced Geothermal Systems
5. **Biomass Energy** — Biogas, anaerobic digestion, gasification, pyrolysis, sustainable feedstocks
6. **Green Hydrogen** — Alkaline, PEM, and SOEC electrolysis, energy demand, industrial decarbonization
7. **Renewable Energy Storage** — LFP & NMC BESS, Vanadium flow batteries, sodium-ion, pumped hydro
8. **Government Renewable Energy Schemes** — PM Surya Ghar, PM-KUSUM, National Green Hydrogen Mission, PLI, US IRA, EU Green Deal

---

## 4. Setup & Installation Instructions

### Step 1: Navigate to Project Directory
```bash
cd renewable-energy-assistant
```

### Step 2: Create & Activate a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (`.env`)
Open `.env` and replace the placeholder with your real Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey):

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=700
CHUNK_OVERLAP=100
TOP_K_RESULTS=4
MIN_SIMILARITY_SCORE=0.32
```

> **Security Note:** Never hardcode API keys in source code or commit `.env` with real secrets to version control.

---

## 5. Running the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Quick Start Walkthrough
1. Open **Home Dashboard** or **Upload Documents** and click **Load & Index Sample Dataset** (or upload your own PDF/TXT files).
2. Navigate to **Ask AI Assistant**, pick an example question or type your own question.
3. Inspect the grounded Gemini response and expand **Source References** to verify the PDF filename, page number, relevance score, and exact retrieved text snippet.
4. Visit **Knowledge Repository** to inspect category statistics, search indexed documents, or delete individual documents.

---

## 6. Testing Questions & Expected Results

| # | Test Question | Expected Source & Page | Expected Behavior |
|---|---|---|---|
| 1 | *What are the efficiencies of monocrystalline, polycrystalline, and perovskite solar cells?* | `01_solar_and_wind_energy_systems.pdf` (Page 1) | Reports 20–23.5% (mono PERC/TOPCon), 15–17% (poly), and >33% lab tandem perovskite-silicon efficiency with citation. |
| 2 | *What is the Betz limit and how does wind speed affect turbine power output?* | `01_solar_and_wind_energy_systems.pdf` (Page 2) | Explains 59.3% (16/27) Betz limit and cubic wind speed relationship ($P \propto v^3$). |
| 3 | *How do Pelton, Francis, and Kaplan turbines differ by hydraulic head?* | `02_hydro_geothermal_biomass_energy.pdf` (Page 1) | Distinguishes Pelton (>300 m), Francis (30–300 m), and Kaplan (<30 m). |
| 4 | *Compare Alkaline, PEM, and SOEC electrolyzers used for green hydrogen.* | `03_green_hydrogen_and_energy_storage.pdf` (Page 1) | Summarizes electrolyte, efficiency/ramp speed, catalyst/temperature differences, and 50–55 kWh/kg requirement. |
| 5 | *Explain the subsidy structure under PM Surya Ghar: Muft Bijli Yojana.* | `04_government_renewable_energy_schemes.pdf` (Page 1) | Lists Rs 30,000/kW (first 2 kW), Rs 18,000 (3rd kW), capped at Rs 78,000, targeting 10M households. |
| 6 | *What is the current stock price of Apple Inc.?* (Out-of-scope test) | None | Clearly informs the user that no relevant information was found in the uploaded documents; does **not** fabricate an answer. |
