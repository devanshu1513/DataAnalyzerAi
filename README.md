# AI-Based Data Analyzer

<img src="https://res.cloudinary.com/b0tb1mho/image/upload/v1784651004/wmrc9tkchwggidppyu66.webp"/>

> An AI-powered data analysis application for interactive Exploratory Data Analysis (EDA), intelligent dataset querying, automated visualization, and natural-language data analysis.

## Overview

DataAnalyzerAi transforms spreadsheet analysis into an interactive AI-powered experience.

Users can upload CSV or Excel datasets and explore their data through automated Exploratory Data Analysis (EDA), dataset profiling, spreadsheet viewing, pair plot visualization, and a natural-language chatbot.

The application uses a **hybrid data analysis architecture**:

- Direct Pandas-based analysis for questions that require numerical or categorical calculations.
- FAISS-based Retrieval-Augmented Generation (RAG) for broader dataset-related questions.
- G4F for generating natural-language explanations and responses.

This architecture allows numerical results to be calculated directly from the uploaded dataset while still providing AI-powered conversational analysis.

---

## My Contributions

This version is a **substantially modified and extended implementation** of the original DataAnalyzerAi project.

The major changes and improvements include:

- Redesigned the Streamlit user interface and overall application layout.
- Improved the presentation of the **data profile, Excel/spreadsheet view, and pair plot analysis**.
- Improved the overall chatbot interaction and user experience.
- Added and refined **direct Pandas-based analytical query handling**.
- Integrated deterministic dataset calculations with the RAG-based analysis workflow.
- Improved the organization and presentation of analysis results and visualizations.
- Optimized the RAG workflow by processing the profile report into smaller chunks before retrieval.
- Improved the overall application workflow to make dataset exploration easier through a single interface.

The original project provided the foundation for the data-analysis and RAG functionality, while this version focuses on extending the interface, workflow, analysis experience, and user interaction.

---

## Features

- Upload and analyze **CSV** and **Excel (XLS/XLSX)** datasets.
- Interactive spreadsheet viewer for exploring uploaded data.
- Automated Exploratory Data Analysis (EDA).
- Comprehensive data profiling using YData Profiling.
- Downloadable data profiling report.
- Interactive pair plot generation for numerical features.
- Natural-language dataset querying.
- Direct Pandas-based calculations for supported analytical questions.
- FAISS-powered vector database for semantic retrieval.
- Sentence Transformer embeddings.
- Retrieval-Augmented Generation (RAG) for broader dataset questions.
- G4F-powered natural-language responses.
- Chunked profile-report processing to reduce excessive prompt size.
- Streamlit-based interactive interface.
- Natural-language querying without requiring SQL or Python for basic analysis.

---

## How It Works

DataAnalyzerAi uses two complementary analysis paths depending on the type of question.

### Direct Dataset Analysis

Questions that require an exact calculation are processed directly using Pandas.

```text
User Question
     │
     ▼
analyze_dataset()
     │
     ▼
Can the question be calculated directly?
     │
     ├────────────── YES
     │
     ▼
Pandas DataFrame
     │
     ▼
Calculate Result
     │
     ▼
    G4F
     │
     ▼
Natural Language Explanation
```

This approach ensures that numerical answers are calculated directly from the dataset instead of relying on the language model to perform the calculation.

### 2. RAG-Based Analysis

Questions that are not handled by the direct-analysis layer are passed to the retrieval pipeline.

```text
Uploaded Dataset
    ↓
    Pandas
    ↓
YData Profiling
    ↓
Profile Report
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Database
    ↓
User Question
    ↓
Relevant Chunks Retrieved
    ↓
    G4F
    ↓
Natural Language Response
```

### Complete Workflow

```text
                    Upload CSV / Excel
                           │
                           ▼
                     Load Dataset
                      with Pandas
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
         YData Profiling        Raw DataFrame
                │                     │
                ▼                     ▼
           Profile Text          Direct Analysis
                │                with Pandas
                ▼                     │
             Chunking                 │
                │                     │
                ▼                     │
        Sentence Transformer          │
            Embeddings                │
                │                     │
                ▼                     │
              FAISS                   │
                │                     │
                └──────────┬──────────┘
                           │
                           ▼
                      User Question
                           │
                           ▼
                    analyze_dataset()
                           │
                       Calculation Needed
                       ┌──────┴──────┐
                       │             │
                     YES             NO
                       │             │
                       ▼             ▼
                     Pandas         FAISS
                     Result       Retrieval
                       │             │
                       └──────┬──────┘
                              │
                              ▼
                             G4F
                              │
                              ▼
                      Natural Language Answer
```

---

## Example Questions

The chatbot can answer questions such as:

### Direct Calculation

**Question:**

> What is the average monthly charge of customers who churned?

The application calculates the value directly from the dataset using Pandas and returns:

```text
Average MonthlyCharges of churned customers: $74.44
```

### Group-Based Analysis

**Question:**

> Which contract type has the highest churn rate?

The application calculates the churn rate for each contract type and identifies the highest one.

```text
Month-to-month contracts have the highest churn rate at 42.71%.
```

### General Dataset Questions

Questions that cannot be handled by the direct calculation layer are processed through the FAISS retrieval pipeline and provided to G4F with relevant context.

---

## Why Hybrid Analysis?

A purely LLM-based approach can produce incorrect numerical calculations or consume a large amount of context.

This application therefore separates the problem into two paths:

### Deterministic Analysis

```text
Question → Pandas → Exact Dataset Calculation → G4F Explanation
```

Pandas performs the actual calculation, while G4F is used only to explain the result.

### Retrieval-Based Analysis

```text
Question → FAISS → Relevant Context → G4F → Response
```

This allows the application to handle broader questions while keeping the amount of information sent to the language model under control.

---

## RAG Optimization

The original profiling workflow could produce a very large profile report.

Sending the complete report to the language model can result in excessive prompt size and API/provider failures.

To address this, the profile report is divided into smaller overlapping chunks before embedding.

```text
Large Profile Report
      ↓
RecursiveCharacterTextSplitter
      ↓
Multiple Text Chunks
      ↓
Sentence Transformer
      ↓
FAISS
```

The chatbot retrieves only the most relevant chunks for a user's question.

The current retrieval configuration uses the top **3 relevant chunks**.

This significantly reduces the amount of context passed to the G4F model.

---

## Technology Stack

### Frontend

- Streamlit

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Data Profiling

- YData Profiling

### AI / NLP

- G4F
- Sentence Transformers
- LangChain Text Splitters

### Vector Search

- FAISS

### Visualization

- Matplotlib
- Seaborn

### Supporting Libraries

- BeautifulSoup
- Base64
- Python standard library

---

## Application Workflow

### Step 1 — Upload Dataset

Upload a CSV or Excel file through the Streamlit interface.

### Step 2 — Dataset Loading

The uploaded dataset is loaded into a Pandas DataFrame.

### Step 3 — Exploratory Data Analysis

The application provides access to:

- Dataset preview
- Data profiling
- Numerical feature analysis
- Pair plots

### Step 4 — Profile Generation

YData Profiling generates a detailed report containing information about the dataset.

### Step 5 — Profile Processing

The generated profile is converted into text and divided into smaller chunks.

### Step 6 — Embedding Generation

Each chunk is converted into a numerical vector using a Sentence Transformer model.

### Step 7 — FAISS Indexing

The generated embeddings are stored in a FAISS vector database.

### Step 8 — User Query

The user asks a question in natural language.

### Step 9 — Query Routing

The application checks whether the question can be answered through direct Pandas analysis.

- If yes → Pandas calculates the result.
- If no → FAISS retrieves relevant context.

### Step 10 — AI Explanation

G4F converts the calculated result or retrieved context into a natural-language response.

---

## Installation

Clone your repository:

```bash
git clone https://github.com/devanshu1513/DataAnalyzerAi.git
cd DataAnalyzerAi
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Usage

1. Start the Streamlit application.
2. Upload a CSV or Excel dataset.
3. Explore the uploaded data.
4. Generate and inspect the profiling report.
5. Explore numerical relationships using pair plots.
6. Ask questions about the dataset through the chatbot.
7. The application determines whether the question can be calculated directly.
8. Pandas performs supported calculations.
9. For broader questions, FAISS retrieves relevant context.
10. G4F generates the final natural-language response.

---

## Project Structure

```text
DataAnalyzerAi/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
└── images/
```

---

## Key Implementation Highlights

### Direct Data Analysis

The application uses Pandas for deterministic calculations such as:

- Churn counts
- Churn percentages
- Average monthly charges
- Contract-wise churn rates
- Other supported dataset-specific calculations

### Semantic Retrieval

The YData profile report is divided into smaller chunks before embedding.

This enables FAISS to retrieve only the most relevant sections instead of passing the entire profile report to the language model.

### AI Response Generation

G4F is used as the natural-language generation layer.

The model receives either:

- A calculated Pandas result, or
- Relevant context retrieved from FAISS.

This separation keeps numerical computation independent from language generation.

---

## Future Improvements

Possible extensions include:

- Support for more natural-language statistical queries
- Automatic visualization generation based on user questions
- SQL-based dataset querying
- More advanced statistical analysis
- Automatic feature engineering suggestions
- Additional machine-learning model recommendations
- Support for multiple datasets in a single session
- Improved query classification
- More advanced RAG retrieval strategies
- Interactive dashboard generation

---

## Acknowledgements

This project is a substantially modified and extended version of the open-source project **DataAnalyzerAi** by **thebitanpaul**.

Original repository:

[github.com/thebitanpaul/DataAnalyzerAi](https://github.com/thebitanpaul/DataAnalyzerAi)

The original project provided the foundation for the data-analysis and RAG application. This version has been substantially modified, including changes to the **user interface, application workflow, data-analysis functionality, chatbot behavior, visualization layout, and overall user experience**.

The original project is licensed under the **Apache License 2.0**. The applicable copyright and license notices from the original project are retained in accordance with the license terms.

---

## License

This project incorporates and modifies code from the original **DataAnalyzerAi** project by **thebitanpaul**, which is licensed under the **Apache License 2.0**.

See the `LICENSE` file for the full license text.

---

## Author

**Devanshu Kumar**

B.Tech — Chemical Science and Technology

Indian Institute of Technology Patna

GitHub: `devanshu1513`

---
