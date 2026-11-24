"""Retrieval eval for the nanocoder repo (https://github.com/Nano-Collective/nanocoder).

Every expected path below was checked against rag.files for REPO_ID.
Negative cases were checked with grep over the indexed files: the feature
is not in the repo, so the right result is "I don't know" (top distance
above MAX_DISTANCE).

Categories:
  direct      - question words match the file or function names
  paraphrase  - same intent, different words (tests the embedding)
  detail      - asks about one constant or behaviour inside a file
  identifier  - raw code identifier as the query
  docs        - answer lives in docs/*.md
  ambiguous   - more than one file is a correct answer
  noisy       - typos, Hinglish, one-word, long rambling questions
  neg_offtopic - nothing to do with code
  neg_absent   - software question, but the feature does not exist here
"""

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from main import (CHUNK_SIZE, KEYWORD_WEIGHT, MAX_DISTANCE, OVERLAP, RRF_K, AskRequest,
                  hybrid_search, search_chunk)

REPO_ID = "8f34f915-9c6a-4942-a4bc-bc823c66190f"


class EvalCase(NamedTuple):
    category: str
    question: str
    expected: tuple[str, ...]  # empty tuple = negative case


EVAL_SET: list[EvalCase] = [
    # ---------- direct ----------
    EvalCase("direct", "How are tools registered?", ("source/tools/tool-registry.ts",)),
    EvalCase("direct", "How does voice mode record audio from the microphone?", ("plugins/voice/src/microphone.ts",)),
    EvalCase("direct", "How are cron schedules parsed?", ("source/schedule/cron.ts",)),
    EvalCase("direct", "How are secrets redacted from logs?", ("source/utils/logging/redaction.ts",)),
    EvalCase("direct", "How is the conversation compacted when the context is full?", ("source/conversation/compaction/auto-compact.ts", "source/conversation/compaction/auto-compact-session.ts", "docs/features/context-compression.md")),
    EvalCase("direct", "How is the git commit message prompt built?", ("source/tools/git/commit-prompt.ts",)),
    EvalCase("direct", "How does the MCP client connect to a server?", ("source/mcp/mcp-client.ts",)),
    EvalCase("direct", "How does fuzzy matching work?", ("source/utils/fuzzy-matching.ts",)),
    EvalCase("direct", "How are session titles generated?", ("source/session/title-generator.ts", "source/session/maybe-generate-title.ts")),
    EvalCase("direct", "How does the bash sandbox restrict commands?", ("source/services/bash-sandbox.ts",)),
    EvalCase("direct", "How does the MCP health monitor ping servers and reconnect?", ("source/mcp/health-monitor.ts",)),
    EvalCase("direct", "How are subagent definitions loaded from the agents folder?", ("source/subagents/subagent-loader.ts",)),
    # ---------- paraphrase ----------
    EvalCase("paraphrase", "How does the agent stop the model from fetching internal network addresses like cloud metadata endpoints?", ("source/tools/fetch-url-guard.ts",)),
    EvalCase("paraphrase", "How does it avoid leaving a half-written file if the process crashes during a save?", ("source/utils/fs/atomic-write.ts",)),
    EvalCase("paraphrase", "How does it keep Windows-style line breaks when the model rewrites a file?", ("source/tools/file-ops/line-endings.ts",)),
    EvalCase("paraphrase", "What stops a burst of file change events from launching hundreds of parallel runs?", ("source/events/backpressure.ts",)),
    EvalCase("paraphrase", "How does the terminal app tell a paste apart from fast typing?", ("source/input/paste-detection.ts",)),
    EvalCase("paraphrase", "How does the terminal UI find the running background process, and how is a dead one cleaned up?", ("source/daemon/lockfile.ts",)),
    EvalCase("paraphrase", "How does it find out which language servers are installed on the machine?", ("source/lsp/server-discovery.ts",)),
    EvalCase("paraphrase", "Who decides whether a tool call needs the user to approve it first?", ("source/tools/approval-policy.ts",)),
    EvalCase("paraphrase", "How is personal data in prompts hidden before it is sent to the model provider?", ("source/ai-sdk-client/chat/privacy.ts",)),
    EvalCase("paraphrase", "How does the CLI tell the user that a newer version is available?", ("source/services/update-checker.ts",)),
    # ---------- detail ----------
    EvalCase("detail", "What is the delay before the first retry, and how is the exponential backoff capped?", ("source/utils/backoff.ts",)),
    EvalCase("detail", "Which token formats like GitHub PATs, AWS keys and Slack tokens are masked before posting a PR comment?", ("source/verify/redact-secrets.ts",)),
    EvalCase("detail", "What fallback context window sizes are used for Ollama llama models?", ("source/models/models-dev-client.ts",)),
    EvalCase("detail", "Which files is an unattended CI fix not allowed to change before it is pushed, like workflows and lockfiles?", ("source/daemon/ci-fix-path-guard.ts",)),
    EvalCase("detail", "Which tools are in the minimal and nano tool profiles?", ("source/tools/tool-profiles.ts",)),
    EvalCase("detail", "What is the maximum size of a pasted image and which image formats are accepted?", ("source/input/clipboard-image.ts",)),
    EvalCase("detail", "How long does a session-end lifecycle hook get before it times out?", ("source/services/lifecycle-hooks.ts",)),
    EvalCase("detail", "What damping factor is used when ranking files for the repo map?", ("source/repo-map/index.ts",)),
    EvalCase("detail", "How are ${VAR:-default} environment variable references expanded in the config?", ("source/config/env-substitution.ts",)),
    EvalCase("detail", "What is the default shutdown timeout and which environment variable overrides it?", ("source/utils/shutdown/shutdown-manager.ts",)),
    EvalCase("detail", "How are XML-style tool calls parsed for models without native function calling?", ("source/tool-calling/xml-parser.ts",)),
    # ---------- identifier ----------
    EvalCase("identifier", "assertResolvedPublicHttpUrl", ("source/tools/fetch-url-guard.ts",)),
    EvalCase("identifier", "atomicWriteFileSync", ("source/utils/fs/atomic-write.ts",)),
    EvalCase("identifier", "filterSilenceMarkers BLANK_AUDIO", ("plugins/voice/src/stt.ts",)),
    EvalCase("identifier", "detectLineEnding", ("source/tools/file-ops/line-endings.ts",)),
    EvalCase("identifier", "ExponentialBackoff", ("source/utils/backoff.ts",)),
    EvalCase("identifier", "calculateTokenBreakdown", ("source/usage/calculator.ts",)),
    # ---------- docs ----------
    EvalCase("docs", "How do I configure Ollama as a provider?", ("docs/configuration/providers/ollama.md",)),
    EvalCase("docs", "What keyboard shortcuts are available?", ("docs/features/keyboard-shortcuts.md",)),
    EvalCase("docs", "How do I uninstall nanocoder?", ("docs/getting-started/uninstalling.md",)),
    EvalCase("docs", "How do I add an MCP server in the configuration file?", ("docs/configuration/mcp-configuration.md",)),
    EvalCase("docs", "How do I set up and use voice mode?", ("docs/features/voice-mode.md",)),
    # ---------- ambiguous (any listed path counts) ----------
    EvalCase("ambiguous", "How are secrets and API keys hidden?", ("source/utils/logging/redaction.ts", "source/verify/redact-secrets.ts")),
    EvalCase("ambiguous", "How does login with a GitHub Copilot or ChatGPT subscription work?", ("source/auth/github-copilot.ts", "source/auth/chatgpt-codex.ts", "docs/configuration/providers/github-copilot.md", "docs/configuration/providers/chatgpt-codex.md")),
    EvalCase("ambiguous", "How are checkpoints saved and restored?", ("source/services/checkpoint-manager.ts", "source/commands/checkpoint.tsx", "docs/features/checkpointing.md")),
    EvalCase("ambiguous", "How are chat sessions saved and resumed later?", ("source/session/session-manager.ts", "source/hooks/useSessionAutosave.ts", "source/commands/resume.ts", "docs/features/session-management.md")),
    EvalCase("ambiguous", "How are desktop notifications sent when a task finishes?", ("source/services/notifications.ts", "docs/features/notifications.md")),
    # ---------- noisy ----------
    EvalCase("noisy", "how r tools registerd in tool registery", ("source/tools/tool-registry.ts",)),
    EvalCase("noisy", "bash sandbox commands ko kaise restrict karta hai?", ("source/services/bash-sandbox.ts",)),
    EvalCase("noisy", "context full ho jaye to conversation compact kaise hota hai?", ("source/conversation/compaction/auto-compact.ts", "source/conversation/compaction/auto-compact-session.ts", "docs/features/context-compression.md")),
    EvalCase("noisy", "cron", ("source/schedule/cron.ts",)),
    EvalCase("noisy", "redaction", ("source/utils/logging/redaction.ts", "source/verify/redact-secrets.ts")),
    EvalCase("noisy", "So when the AI wants to run a shell command on my laptop, before it actually runs, is there some jail "
              "like sandbox-exec or bubblewrap that limits what the command can touch outside the project?", ("source/services/bash-sandbox.ts",)),
    # ---------- neg_offtopic ----------
    EvalCase("neg_offtopic", "What is the capital of France?", ()),
    EvalCase("neg_offtopic", "Give me a recipe for butter chicken.", ()),
    EvalCase("neg_offtopic", "Who won the FIFA World Cup in 2022?", ()),
    # ---------- neg_absent (grep found 0 hits in the indexed files) ----------
    EvalCase("neg_absent", "How are user passwords hashed with bcrypt before they are stored?", ()),
    EvalCase("neg_absent", "How does the app send push notifications to phones using Firebase?", ()),
    EvalCase("neg_absent", "How are messages published to a Kafka topic?", ()),
    EvalCase("neg_absent", "How is two-factor authentication with TOTP codes implemented?", ()),
    EvalCase("neg_absent", "How is the UI translated into multiple languages with i18n?", ()),
    EvalCase("neg_absent", "How does the leaderboard rank players by score?", ()),
    EvalCase("neg_absent", "How are emails sent over SMTP?", ()),
]


def run(cases: list[EvalCase], mode: str = "hybrid") -> dict[str, dict]:
    """Run every case, print the report, and return per-category stats."""
    by_cat: dict[str, list[tuple[int | None, float, bool]]] = defaultdict(list)

    for case in cases:
        request = AskRequest(question=case.question)
        if mode == "hybrid":
            # Same as ask_question: gate on the top vector distance, not the fused list.
            snippets, top_dist = hybrid_search(REPO_ID, request)
        else:
            snippets = search_chunk(REPO_ID, request)
            top_dist = snippets[0]["distance"] if snippets else 1.0
        paths = [s["file_path"] for s in snippets]
        gated = top_dist > MAX_DISTANCE  # ask_question would answer "I don't know"

        if case.expected:
            rank = next((i + 1 for i, p in enumerate(paths) if p in case.expected), None)
            ok = rank is not None and not gated
            label = "HIT " if ok else ("GATE" if rank else "MISS")
        else:
            rank = None
            ok = gated
            label = "REJ " if ok else "LEAK"

        by_cat[case.category].append((rank, top_dist, ok))
        print(f"{label} [{case.category}] rank={rank} d={top_dist:.3f} | {case.question[:80]}")
        if not ok:
            print("     got:", paths)

    categories: dict[str, dict] = {}
    print(f"\n{'category':<13} {'n':>4} {'pass':>6} {'recall@5':>9} {'MRR':>5}  (MAX_DISTANCE={MAX_DISTANCE})")
    for cat, rows in by_cat.items():
        n = len(rows)
        passed = sum(ok for _, _, ok in rows)
        hits = sum(r is not None for r, _, _ in rows)
        mrr = sum(1 / r for r, _, _ in rows if r) / n
        categories[cat] = {"n": n, "passed": passed, "hits": hits, "mrr": round(mrr, 3)}
        if cat.startswith("neg_"):
            print(f"{cat:<13} {n:>4} {passed / n:>6.0%} {'-':>9} {'-':>5}")
        else:
            print(f"{cat:<13} {n:>4} {passed / n:>6.0%} {hits / n:>9.0%} {mrr:>5.2f}")

    pos_pass, pos_total, neg_pass, neg_total = totals(categories)
    print(f"\nPositive pass (hit in top 5 AND not gated): {pos_pass}/{pos_total} = {pct(pos_pass, pos_total)}")
    print(f"Negative pass (correctly said I don't know): {neg_pass}/{neg_total} = {pct(neg_pass, neg_total)}")
    return categories


# ---------- history + report ----------

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "eval_results" / "history.jsonl"
REPORT_FILE = BASE_DIR / "EVAL_RESULTS.md"


def pct(part: int, total: int) -> str:
    return f"{part / total:.0%}" if total else "-"


def totals(categories: dict[str, dict]) -> tuple[int, int, int, int]:
    """Return (positive passed, positive total, negative passed, negative total)."""
    pos = [c for name, c in categories.items() if not name.startswith("neg_")]
    neg = [c for name, c in categories.items() if name.startswith("neg_")]
    return (sum(c["passed"] for c in pos), sum(c["n"] for c in pos),
            sum(c["passed"] for c in neg), sum(c["n"] for c in neg))


def git_commit() -> str:
    """Short HEAD sha, with -dirty if tracked code differs from it (results files ignored)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=BASE_DIR, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        sha = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no", "--",
                    ".", ":!eval_results", ":!EVAL_RESULTS.md")
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_cases(path: str) -> list[EvalCase]:
    """Load cases from a JSONL file with category, question and expected fields."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [EvalCase(r["category"], r["question"], tuple(r["expected"])) for r in rows]


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def delta(curr: tuple[int, int], prev: tuple[int, int] | None) -> str:
    """Change in pass rate, in percentage points."""
    if prev is None or not curr[1] or not prev[1]:
        return "-"
    return f"{(curr[0] / curr[1] - prev[0] / prev[1]) * 100:+.1f} pp"


def render_report(history: list[dict]) -> str:
    lines = [
        "# Retrieval Eval Results",
        "",
        "Auto-generated by `evals.py` from `eval_results/history.jsonl`. Do not edit by hand.",
        "",
        "```bash",
        'python evals.py --note "what changed"                              # small set (built into evals.py)',
        'python evals.py --file eval_questions.jsonl --note "what changed"  # full set',
        'python evals.py --mode vector --note "what changed"                # vector search only',
        "```",
        "",
        "- **Mode**: `hybrid` = `hybrid_search` (vector + keyword, RRF), what `ask_question` uses. "
        "`vector` = `search_chunk` only. Rows before the mode column existed are `vector`.",
        "- **Positive pass**: the expected file is in the top 5 results AND the top distance is <= `MAX_DISTANCE`.",
        "- **Negative pass**: the question is about something not in the repo, and the top distance is > `MAX_DISTANCE`, so the API answers \"I don't know\".",
        "- **Recall@5** ignores the distance gate. **MRR** is the mean of 1/rank of the first correct file.",
        "- `backfilled` rows were recorded by hand from earlier runs, before history tracking existed.",
    ]
    datasets = list(dict.fromkeys(r["dataset"] for r in history))
    for ds in datasets:
        runs = [r for r in history if r["dataset"] == ds]
        latest = runs[-1]
        pp, pt, np_, nt = totals(latest["categories"])
        lines += [
            "",
            f"## Dataset: `{ds}` ({latest['n_questions']} questions)",
            "",
            f"Latest run: {latest['date']} | commit `{latest['commit']}` | "
            + " | ".join(f"{k}={v}" for k, v in latest["config"].items()),
            "",
            f"**Positive pass: {pp}/{pt} = {pct(pp, pt)}** | **Negative pass: {np_}/{nt} = {pct(np_, nt)}**",
            "",
            "| Category | n | Pass | Recall@5 | MRR |",
            "|---|---:|---:|---:|---:|",
        ]
        for cat, c in latest["categories"].items():
            if cat.startswith("neg_"):
                lines.append(f"| {cat} | {c['n']} | {pct(c['passed'], c['n'])} | - | - |")
            else:
                lines.append(f"| {cat} | {c['n']} | {pct(c['passed'], c['n'])} | "
                             f"{pct(c['hits'], c['n'])} | {c['mrr']:.2f} |")

        lines += [
            "",
            "### History (newest first)",
            "",
            "| # | Date | Commit | Mode | MAX_DISTANCE | Positive pass | Δ | Negative pass | Δ | Note |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        rows = []
        prev: tuple[tuple[int, int], tuple[int, int]] | None = None
        for i, r in enumerate(runs, start=1):
            pp, pt, np_, nt = totals(r["categories"])
            pos, neg = (pp, pt), (np_, nt)
            note = r.get("note", "").replace("|", "\\|")
            if r.get("backfilled"):
                note = f"{note} (backfilled)".strip()
            rows.append(
                f"| {i} | {r['date']} | `{r['commit']}` | {r['config'].get('mode', 'vector')} | "
                f"{r['config'].get('MAX_DISTANCE', '-')} | "
                f"{pp}/{pt} = {pct(pp, pt)} | {delta(pos, prev[0] if prev else None)} | "
                f"{np_}/{nt} = {pct(np_, nt)} | {delta(neg, prev[1] if prev else None)} | {note} |"
            )
            prev = (pos, neg)
        lines += reversed(rows)
    return "\n".join(lines) + "\n"


def save_run(categories: dict[str, dict], dataset: str, n_questions: int, note: str,
             mode: str) -> None:
    config: dict = {"mode": mode, "MAX_DISTANCE": MAX_DISTANCE,
                    "CHUNK_SIZE": CHUNK_SIZE, "OVERLAP": OVERLAP}
    if mode == "hybrid":
        config |= {"RRF_K": RRF_K, "KEYWORD_WEIGHT": KEYWORD_WEIGHT}
    record = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "dataset": dataset,
        "n_questions": n_questions,
        "config": config,
        "note": note,
        "categories": categories,
    }
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    REPORT_FILE.write_text(render_report(load_history()), encoding="utf-8")
    print(f"\nSaved run to {HISTORY_FILE.relative_to(BASE_DIR)} and updated {REPORT_FILE.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval eval with saved history.")
    parser.add_argument("--file", help="JSONL question file (default: EVAL_SET in this file)")
    parser.add_argument("--note", default="", help="What changed since the last run")
    parser.add_argument("--no-save", action="store_true", help="Only print, do not update history")
    parser.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid",
                        help="hybrid_search (what ask_question uses) or search_chunk only")
    args = parser.parse_args()

    cases = load_cases(args.file) if args.file else EVAL_SET
    dataset = Path(args.file).name if args.file else "evals.py"
    categories = run(cases, args.mode)
    if not args.no_save:
        save_run(categories, dataset, len(cases), args.note, args.mode)


if __name__ == "__main__":
    main()
