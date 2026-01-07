# 📰 UPSC AI-Sentinel: Semantic News Intelligence Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://upsc-ai-sentinel.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**UPSC AI-Sentinel** is a sophisticated NLP-powered news aggregator and classifier specifically designed for Civil Services aspirants. It solves the "information overload" problem by semantically filtering thousands of daily news articles into high-yield UPSC syllabus categories using state-of-the-art Deep Learning models.

---

## 🚀 Key Features

- **Semantic Classification**: Uses the `all-mpnet-base-v2` transformer model to categorize news based on *context and meaning*, not just keywords.
- **Trusted Indian Sources**: Specifically focuses on elite sources like *The Hindu, Indian Express, PIB, and LiveMint*.
- **Intelligent Summarization**: Implements an custom extraction-based summarization logic that delivers 2-paragraph insights directly from the source text.
- **Syllabus-Aligned Filtering**: Multi-label classification across Polity, Economy, IR, Environment, S&T, Security, and more.
- **Noise Reduction**: Advanced negative-keyword filtering to eliminate sports, entertainment, and local crime gossip.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Modern, Interactive Dashboard)
- **NLP Models**: Sentence-Transformers (`all-mpnet-base-v2` via HuggingFace)
- **Data Acquisition**: NewsAPI & Newspaper3k (Full article parsing)
- **Logic**: Python (PyTorch for tensor operations)

## 🧠 Technical Deep Dive

Unlike basic news apps, **UPSC AI-Sentinel** calculates a **Relevance Score** for every article. 

1. **Embedding Generation**: Every article title and description is converted into a 768-dimensional vector space.
2. **Cosine Similarity**: The engine compares article vectors against "Syllabus Sub-Profiles" (rich semantic descriptions of UPSC topics).
3. **Tiered Weighted Scoring**: 
    - **Core Keywords**: High boost (0.1 weight)
    - **Supporting Keywords**: Low boost (0.04 weight)
    - **Current Affairs Triggers**: Dynamic boost for "policy launched", "treaty signed", etc.
4. **Classification Engine**: Only articles clearing a specific similarity threshold are presented, ensuring high-quality content only.

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/assistantgoogle/UPSC-AI-Sentinel.git
   cd UPSC-AI-Sentinel
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   NEWSAPI_KEY=your_actual_newsapi_key_here
   ```

4. **Run the application**:
   ```bash
   streamlit run App.py
   ```

## 📈 Future Roadmap

- [ ] Support for **Regional Language** news filtering.
- [ ] Integration with **Google Gemini/OpenAI** for abstractive summarization.
- [ ] Automated **PDF Generation** for weekly/monthly current affairs compilations.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.

---
**Developed with ❤️ for UPSC Aspirants.**
