"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保ったまま続けて質問する（exit で終了）

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）

LLMは前の会話を覚えていない。--chat では履歴を手元のリストに貯めておき、
毎回の質問に直近 MAX_TURNS 往復分の履歴を付けて送り直している。
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

# LLMに送る履歴の上限（直近の往復数。1往復 = 質問1つ + 回答1つ）
MAX_TURNS = 5


def ask(
    client: genai.Client, question: str, history: list[types.Content] | None = None
) -> str:
    # 直近 MAX_TURNS 往復分の履歴の後ろに、今回の質問を付けて送る
    recent = (history or [])[-MAX_TURNS * 2 :]
    question_content = types.Content(role="user", parts=[types.Part(text=question)])
    response = client.models.generate_content(
        model=MODEL, contents=[*recent, question_content]
    )

    # 入力・出力トークン数を表示する（思考するモデルでは思考トークンも別に数えられる）
    usage = response.usage_metadata
    print(f"[送った履歴] {len(recent) // 2} 往復")
    print(f"[入力トークン] {usage.prompt_token_count}")
    print(f"[出力トークン] {usage.candidates_token_count}")
    print(f"[思考トークン] {usage.thoughts_token_count}")
    print(f"[合計トークン] {usage.total_token_count}\n")

    # 呼び出しが成功したときだけ履歴に残す（失敗時に user が2つ続かないように）
    if history is not None:
        history.append(question_content)
        history.append(
            types.Content(role="model", parts=[types.Part(text=response.text)])
        )

    return response.text


def chat(client: genai.Client) -> None:
    history: list[types.Content] = []
    print("会話を始めます（exit で終了）\n")
    while True:
        try:
            question = input("あなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question in ("exit", "quit"):
            break
        if not question:
            continue
        try:
            print(f"\nAI> {ask(client, question, history)}\n")
        except errors.APIError as e:
            # 会話は続けられるように、エラーを表示して次の入力を待つ
            print(api_error_message(e.code, e.message) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--chat", action="store_true", help="会話履歴を保ったまま続けて質問する"
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    client = genai.Client(api_key=get_api_key())
    if args.chat:
        chat(client)
        return

    print(f"[質問] {args.question}\n")
    try:
        print(ask(client, args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
