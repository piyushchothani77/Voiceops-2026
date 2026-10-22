# VoiceOps 2026 — ZentroMart Voice Support Agent

A voice-enabled customer support agent for ZentroMart, built as part of the VoiceOps 2026 capstone.

## Overview

The project combines a language model, retrieval-augmented generation (RAG), and Gnani speech services to support spoken customer queries about orders, cancellations, returns, and refunds.

## Features

- Speech-to-text using Gnani STT.
- Policy-grounded responses using document retrieval.
- Text-to-speech using Gnani TTS.
- Support for follow-up questions in a conversation.
- Retrieved document chunks to help trace the sources of responses.

## Technology Stack

- Python
- Groq API for language-model access
- Transformers and Sentence Transformers
- Gnani Vachana SDK for speech processing
- Pandas and NumPy

## Repository Contents

- `VoiceOps.ipynb` — project notebook and experiments.
- `requirements.txt` — Python dependencies.

Additional runnable scripts, configuration examples, evaluation results, and demonstration materials will be added as the project is packaged.

## Setup

1. Clone this repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure the required API credentials as environment variables before running the notebook. Never commit API keys or secrets to the repository.

## Running

Open `VoiceOps.ipynb` in Jupyter or Google Colab and execute the cells in order. Some steps may require API credentials and access to the project dataset.

## Security

Use environment variables for API credentials. Do not commit `.env` files, real API keys, or customer-sensitive information.

## Project

**VoiceOps 2026 Capstone** — ZentroMart voice-enabled customer support agent.
