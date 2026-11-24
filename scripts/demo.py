"""End-to-end demo against a running stack (docker compose up -d --build).

    python scripts/demo.py [--api http://localhost:8000]

Ingests examples/demo_brand_notes.md for tenant "demo", prints the extracted
facts, then asks two questions. Uses only the standard library.
"""
import argparse
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent


def call(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:  # CPU inference can be slow
        return json.loads(resp.read() or b"{}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--tenant", default="demo")
    args = parser.parse_args()
    base = f"{args.api}/tenants/{args.tenant}"

    text = (ROOT / "examples" / "demo_brand_notes.md").read_text(encoding="utf-8")
    print("Ingesting demo notes and extracting facts with the local LLM...")
    doc = call("POST", f"{base}/documents", {"text": text, "source": "demo_brand_notes.md"})
    print(f"document_id={doc['document_id']} chunks={doc['chunks']} facts={len(doc['facts'])}")
    for f in doc["facts"]:
        print(f"  [{f['category']}] {f['fact']}")

    questions = [
        ("Write three reel hooks for next week.", ["tone_of_voice", "target_audience"]),
        ("What should my lead magnet funnel look like this quarter?", ["products", "goals"]),
    ]
    for question, categories in questions:
        print(f"\nQ: {question}")
        res = call("POST", f"{base}/ask", {"question": question, "required_categories": categories})
        print(f"A: {res['answer']}\n  ({len(res['sources'])} sources)")


if __name__ == "__main__":
    main()
