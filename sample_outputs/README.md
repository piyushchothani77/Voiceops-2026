# Sample outputs

Running `run.py` writes one JSON record per audio input, a combined `results.json`, and synthesized WAV replies in `audio/`.

Example command:

```bash
python run.py --input_dir path/to/audio --output_dir sample_outputs --language hi-IN
```

The output JSON contains `input_audio`, `customer_text`, `reply`, `reply_audio`, and `retrieved_chunks`. Actual outputs depend on the audio and current knowledge-base data. Do not commit private recordings or API credentials.
