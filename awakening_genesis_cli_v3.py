# -*- coding: utf-8 -*-
"""
Awakening Genesis Dynamics CLI v3.0
-----------------------------------
役割：
  - Awakening5XOS_V3 を起動（OS 自身が GeminiBackend を内部で確保）
  - my_soul.json をロード／セーブ
  - AwakeningGenesisEngine v3 を attach
  - REPL で通常対話 & /flow コマンドを扱う
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np

from awakening5x_os_v3 import Awakening5XOS_V3
from awakening_genesis_engine_v3 import AwakeningGenesisEngine

SOUL_PATH = "my_soul.json"


# ============================================================
# Utility: soul ファイル I/O
# ============================================================

def soul_exists() -> bool:
    return os.path.exists(SOUL_PATH)


def banner() -> None:
    print("===========================================")
    print("   🧠 Awakening Genesis Dynamics CLI v3.0")
    print("===========================================\n")


# ============================================================
# メイン
# ============================================================

def main() -> None:
    banner()

    # ---------- Awakening5X OS を起動 ----------
    print("[System] Booting Awakening5XOS_V3...")

    try:
        os_core = Awakening5XOS_V3()  # ★ backend=None → OS が内部で GeminiBackend を import
    except Exception as e:
        print("[System] Fatal: Awakening5XOS_V3 の初期化に失敗しました。")
        print("  Error:", repr(e))
        sys.exit(1)

    # ---------- 魂ファイルのロード ----------
    if soul_exists():
        try:
            os_core.load_state(SOUL_PATH)
            print(f"[Soul] Restored state from {SOUL_PATH} (step={os_core.step})")
        except Exception as e:
            print(f"[Soul] Warning: {SOUL_PATH} の読み込みに失敗しました:", e)
    else:
        print("[Soul] First boot: no soul file found.")

    # ---------- Teleology ゴールの表示 ----------
    try:
        goals = os_core.goal_labels()
    except Exception:
        # goal_labels() がもし無ければ Teleology から直接取る
        goals = [g["name"] for g in getattr(os_core.teleology, "goals", [])]
    print(f"\n[System] Checking Teleology goals...")
    print(f"[System] Active goals: {goals}\n")

    # ---------- Genesis エンジン起動 & attach ----------
    try:
        genesis = AwakeningGenesisEngine(dim=4)
        genesis.attach(os_core)
        print("[System] Initializing Genesis Dynamics Engine (dim=4)...")
        print("[System] Genesis ready. Use `/flow <text>` to see trajectories.\n")
    except Exception as e:
        print("[System] Warning: Genesis エンジンの初期化に失敗しました:", e)
        genesis = None

    print("--- Link established. Type text to talk. `exit` or `quit` to leave. ---\n")

    # ---------- REPL ループ ----------
    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[System] Shutting down...")
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower in ("exit", "quit"):
            break

        # ---------- /flow コマンド ----------
        if user_input.startswith("/flow"):
            if genesis is None:
                print("[Flow] Genesis engine is not available.")
                continue

            text = user_input[len("/flow"):].strip()

            # /flow だけなら最後の user 発話を流用
            if not text:
                last_user = None
                for role, msg in reversed(history):
                    if role == "user":
                        last_user = msg
                        break
                if last_user is None:
                    print("[Usage] /flow <text>  （もしくは直前のユーザー発話が必要）")
                    continue
                text = last_user
                print(f"[Genesis] Using last user text as intent: {text!r}")

            print(f"[Flow] Running Genesis dynamics for: {text!r}")

            try:
                # Awakening5XOS_V3 の manifold introspection を使ってベクトル化
                mani_info = os_core.compute_manifold_for_text(text)
                vec = np.array(mani_info["vector"], dtype=float)

                # Genesis に流す
                result = genesis.run_flow(vec)
                traj = result.get("trajectory", [])
                steps = result.get("steps", len(traj))

                print(f"[Flow] Steps: {steps}")
                if traj:
                    print(f"[Flow] Initial[0:4]: {traj[0][:4]}")
                    print(f"[Flow] Final  [0:4]: {traj[-1][:4]}")
                print("[Flow] Done.\n")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Flow] Error during simulation: {e}\n")

            continue  # /flow はここで終わり

        # ---------- 通常対話 ----------
        try:
            # history を渡して Teleology リランキング付き応答を生成
            result = os_core.generate_guided_reply(
                history=history,
                user_text=user_input,
                num_candidates=3,
            )
            reply = result.get("reply") or result.get("best_reply") or ""
            score = float(result.get("score", 0.0))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Error] Failed to generate reply: {e}\n")
            continue

        print(f"\nGemini > {reply}")
        print(f"[Meta] Teleology Score: {score:.4f}\n")

        # 履歴更新（最低限）
        history.append(("user", user_input))
        history.append(("assistant", reply))

        # 魂のセーブ
        try:
            os_core.save_state(SOUL_PATH)
            # print(f"[Soul] Saved to {SOUL_PATH}.\n")
        except Exception as e:
            print(f"[Soul] Warning: save_state 失敗: {e}")

    # ---------- 終了時の最終セーブ ----------
    try:
        os_core.save_state(SOUL_PATH)
        print(f"[System] Final soul saved to {SOUL_PATH}. Goodbye.")
    except Exception as e:
        print(f"[System] Warning: final save_state 失敗: {e}")


if __name__ == "__main__":
    main()
