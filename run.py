import argparse
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient

from src.agent import support_turn_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="sample_outputs")
    parser.add_argument("--language", default="hi-IN")
    args = parser.parse_args()

    load_dotenv()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    audio_output = output_dir / "audio"
    audio_output.mkdir(parents=True, exist_ok=True)

    stt = GnaniSTTClient()
    tts = GnaniTTSClient()

    audio_files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}
    ]

    if not audio_files:
        raise SystemExit(f"No supported audio files found in {input_dir}")

    results = []

    for audio_path in audio_files:
        session_id = str(uuid.uuid4())
        print(f"Processing: {audio_path.name}")

        try:
            transcription = stt.transcribe(
                str(audio_path), language_code=args.language
            )
            customer_text = transcription["transcript"].strip()

            agent_result = support_turn_v2(
                session_id=session_id,
                user_message=customer_text,
            )
            reply = agent_result["reply"]

            reply_audio = audio_output / f"{audio_path.stem}_reply.wav"
            tts.synthesize(
                reply,
                voice="Kaveri",
                model="timbre-v2.5",
                language=args.language,
                output_file=str(reply_audio),
            )

            record = {
                "input_audio": audio_path.name,
                "customer_text": customer_text,
                "reply": reply,
                "reply_audio": str(reply_audio),
                "retrieved_chunks": agent_result.get("retrieved_chunks", []),
            }
            results.append(record)

            (output_dir / f"{audio_path.stem}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved reply: {reply_audio}")

        except Exception as exc:
            print(f"Failed on {audio_path.name}: {exc}")
            results.append({"input_audio": audio_path.name, "error": str(exc)})

    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
