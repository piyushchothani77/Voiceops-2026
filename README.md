# VoiceOps 2026 — ZentroMart Voice Support Agent

Repo link:-https://github.com/piyushchothani77/Voiceops-2026

A voice-enabled customer-support agent built for the VoiceOps 2026 capstone. It combines Gnani speech-to-text (STT), policy retrieval (RAG), a Groq-hosted language model, and Gnani text-to-speech (TTS).

## Features

- Transcribes customer audio with Gnani STT.
- Retrieves relevant ZentroMart knowledge-base policy excerpts.
- Generates concise responses in the customer's language.
- Synthesizes spoken responses with Gnani TTS.
- Saves a JSON record for each input, including transcript, reply, audio path, and retrieved evidence.
- Includes order-verification and support-ticket flows in the agent.

## Repository layout

- `VoiceOps.ipynb` — development notebook.
- `src/agent.py` — retrieval and customer-support agent.
- `run.py` — batch audio CLI.
- `requirements.txt` — Python dependencies.
- `.env.example` — names of required environment variables (no secrets).
- `sample_outputs/` — a representative JSON example and output instructions.
- `report.pdf` — detailed project report covering the design, implementation, testing, limitations, and next steps.
-`demo.mp4` — summarative video examples

## Requirements

- Python 3.10+ recommended.
- The capstone dataset extracted locally.
- Gnani STT/TTS access and a Groq API key.

## Setup

1. Clone the repository and enter the project directory.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Extract the supplied capstone dataset. Set `VOICEOPS_DATA_DIR` to the extracted `Capstone_Project_Dataset_FTTT_2026` directory. It must contain the `data/`, `kb/`, and `tools/` folders.
4. Copy `.env.example` to `.env` and fill in credentials locally:

   ```dotenv
   GROQ_API_KEY=your_groq_key
   GNANI_API_KEY=your_gnani_key
   VOICEOPS_DATA_DIR=/absolute/path/to/Capstone_Project_Dataset_FTTT_2026
   ```

   Never commit `.env`, real credentials, or the dataset archive.

## Run the audio pipeline

Place supported audio files in an input directory. The CLI currently accepts `.wav`, `.mp3`, `.m4a`, and `.flac`:

```bash
python run.py --input_dir path/to/audio --output_dir sample_outputs --language hi-IN
```

For English audio, use `--language en-IN` if supported by your Gnani account. If your source is an MP4 video, extract its audio to WAV first; MP4 is not currently accepted directly by `run.py`.

The output directory contains per-file JSON records, `results.json`, and an `audio/` directory with generated WAV replies.

## Notes and limitations

- Retrieval depends on the provided knowledge base; responses should not be treated as evidence beyond the retrieved excerpts.
- The agent uses the official dataset order/ticket tools and should not disclose order details before verification.
- A successful TTS call and a playable output file should be checked when validating an environment.
- This repository does not include the dataset, local environment, or API credentials.

## Security

Do not commit `.env`, API keys, the dataset ZIP, virtual environments, caches, or customer recordings.

## Capstone

VoiceOps 2026 — ZentroMart voice-enabled customer-support agent.
