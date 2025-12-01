"""
Awakening5X OS v2.0 – Gemini 2.5 Flash Edition
----------------------------------------------
Persistent 5-Axis Cognitive OS for stateless LLM APIs (Gemini).

Axes:
  - Manifold  : 64-dim projected semantic manifold
  - Identity  : persistent entity store with activations
  - Causal    : directed weighted edges between entities
  - Essence   : archetype basis on manifold
  - Teleology : goal vectors, used to re-rank candidate replies

主要な公開インターフェイス:
  - GeminiBackend         : Gemini API ラッパー
  - Awakening5XOS         : 5軸OS本体
      * observe_text(...)
      * add_goal_from_text(...)
      * generate_guided_reply(...)
      * manifold_report(text)         # CLI の /mani 用 dict
      * compute_manifold_for_text(...)

      * identity_snapshot(top_k=...)  # /id 用
      * causal_snapshot(top_k=...)    # /causal 用

      * save_state(path="my_soul.json")
      * load_state(path="my_soul.json")

依存:
    pip install numpy google-genai
環境変数:
    GOOGLE_API_KEY または GEMINI_API_KEY
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# ================================================================
#                         Utility
# ================================================================

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = x.astype(float)
    x = x - np.max(x)
    ex = np.exp(x / max(temp, 1e-8))
    s = np.sum(ex)
    if s <= 0:
        return np.zeros_like(ex)
    return ex / s


# ================================================================
#                     1. Identity / Entities
# ================================================================

class EntityType(Enum):
    UNKNOWN = auto()
    PERSON  = auto()
    OBJECT  = auto()
    CONCEPT = auto()
    EVENT   = auto()


@dataclass
class EntityNode:
    eid: int
    name: str
    type_tag: EntityType
    vector: np.ndarray
    activation: float = 1.0
    tags: List[str] = field(default_factory=list)


@dataclass
class IdentityStore:
    """
    永続化される「存在レジストリ」。
    - entities: eid(str) -> EntityNode
    - name_to_id: name.lower() -> eid(int)
    """
    entities: Dict[str, EntityNode] = field(default_factory=dict)
    name_to_id: Dict[str, int] = field(default_factory=dict)
    next_eid: int = 0

    def step_decay(self, rate: float = 0.97) -> None:
        for e in self.entities.values():
            e.activation *= rate

    def get_or_create(
        self,
        name: str,
        vec: np.ndarray,
        etype: EntityType = EntityType.UNKNOWN,
    ) -> EntityNode:
        key = name.lower().strip()
        if not key:
            key = "<EMPTY>"

        if key in self.name_to_id:
            eid = self.name_to_id[key]
            node = self.entities[str(eid)]
            # 埋め込みを少しだけ更新
            node.vector = 0.8 * node.vector + 0.2 * vec
            node.activation = 1.0
            return node

        eid = self.next_eid
        self.next_eid += 1
        node = EntityNode(eid=eid, name=name, type_tag=etype, vector=vec)
        self.entities[str(eid)] = node
        self.name_to_id[key] = eid
        return node

    def top_entities(self, top_k: int = 32) -> List[Dict[str, Any]]:
        ents = list(self.entities.values())
        ents.sort(key=lambda e: e.activation, reverse=True)
        out: List[Dict[str, Any]] = []
        for e in ents[:top_k]:
            out.append(
                {
                    "eid": int(e.eid),
                    "type": e.type_tag.name,
                    "activation": float(e.activation),
                    "name": e.name,
                }
            )
        return out


# ================================================================
#                       2. Causal Graph (v2)
# ================================================================

@dataclass
class CausalGraph:
    """
    v2: edges の値は dict:
        {
          "weight": float,
          "count": int,
          "label": Optional[str]
        }
    古い float 形式も自動的に dict にラップする（後方互換）。
    """
    edges: Dict[str, dict] = field(default_factory=dict)
    decay_rate: float = 0.985

    # ---- internal helper -------------------------------------------------
    def _ensure_edge(self, src: int, tgt: int) -> Tuple[str, dict]:
        key = f"{src}->{tgt}"
        rec = self.edges.get(key)

        # すでに dict 形式
        if isinstance(rec, dict):
            return key, rec

        # float -> dict に変換
        if rec is None:
            w = 0.0
        else:
            try:
                w = float(rec)
            except Exception:
                w = 0.0

        rec = {"weight": w, "count": 0, "label": None}
        self.edges[key] = rec
        return key, rec

    # ---- register --------------------------------------------------------
    def register(
        self,
        src: Optional[int],
        tgt: Optional[int],
        weight: float,
        label: Optional[str] = None,
    ) -> None:
        if src is None or tgt is None or src == tgt:
            return

        key, rec = self._ensure_edge(src, tgt)

        old_w = float(rec.get("weight", 0.0))
        new_w = 0.9 * old_w + 0.1 * float(weight)

        rec["weight"] = new_w
        rec["count"] = int(rec.get("count", 0)) + 1

        if label is not None:
            rec["label"] = label

        self.edges[key] = rec

    # ---- decay -----------------------------------------------------------
    def decay(self) -> None:
        dead = []
        for key, rec in list(self.edges.items()):

            # dict 形式
            if isinstance(rec, dict):
                w = float(rec.get("weight", 0.0)) * self.decay_rate
                rec["weight"] = w
                self.edges[key] = rec

            # 古い float 形式（互換モード）
            else:
                try:
                    w = float(rec) * self.decay_rate
                except Exception:
                    w = 0.0
                self.edges[key] = {
                    "weight": w,
                    "count": 0,
                    "label": None,
                }

            if abs(self.edges[key]["weight"]) < 0.01:
                dead.append(key)

        for k in dead:
            del self.edges[k]

    # ---- snapshot --------------------------------------------------------
    def top_edges(self, k: int = 10) -> List[Tuple[int, int, float, Optional[str], int]]:
        items: List[Tuple[int, int, float, Optional[str], int]] = []
        for key, rec in self.edges.items():
            try:
                s, t = key.split("->")
                s_i, t_i = int(s), int(t)

                if isinstance(rec, dict):
                    w = float(rec.get("weight", 0.0))
                    label = rec.get("label")
                    count = int(rec.get("count", 0))
                else:
                    # old float mode
                    w = float(rec)
                    label = None
                    count = 0

                items.append((s_i, t_i, w, label, count))
            except Exception:
                continue

        items.sort(key=lambda x: abs(x[2]), reverse=True)
        return items[:k]


# ================================================================
#                        3. Essence
# ================================================================

@dataclass
class EssenceSpace:
    """
    Manifold 上の「8つの原型ベクトル」。
    文章ベクトルを更新し続けることで、自然と「よく出るパターン」を学習する。
    """
    dim: int
    num_archetypes: int = 8
    lr: float = 0.08
    archetypes: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(1234)
        raw = rng.normal(0, 1, (self.num_archetypes, self.dim))
        norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
        self.archetypes = raw / norms

    def update(self, vec: np.ndarray) -> np.ndarray:
        v = vec / (np.linalg.norm(vec) + 1e-8)
        sims = self.archetypes @ v
        idx = int(np.argmax(sims))
        diff = v - self.archetypes[idx]
        self.archetypes[idx] += self.lr * diff
        self.archetypes[idx] /= np.linalg.norm(self.archetypes[idx]) + 1e-8
        return sims  # 各 archetype への類似度


# ================================================================
#                       4. Teleology
# ================================================================

@dataclass
class Teleology:
    dim: int
    goals: List[Dict[str, Any]] = field(default_factory=list)

    def add_goal(self, name: str, description: str, vec: np.ndarray, weight: float = 1.0) -> None:
        v = vec / (np.linalg.norm(vec) + 1e-8)
        self.goals.append(
            {
                "name": name,
                "description": description,
                "vec": v,
                "weight": float(weight),
            }
        )

    def alignment(self, vec: np.ndarray) -> Dict[str, float]:
        if not self.goals:
            return {}
        v = vec / (np.linalg.norm(vec) + 1e-8)
        scores: Dict[str, float] = {}
        for g in self.goals:
            sim = float(np.dot(v, g["vec"]))
            scores[g["name"]] = float(g["weight"]) * sim
        return scores


# ================================================================
#                     5. Manifold Projector
# ================================================================

class ManifoldProjector:
    """
    d_embed → d_mani への写像。
    簡易なランダム射影 + 非線形正規化。
    """

    def __init__(self, d_in: int, d_mani: int = 64, seed: int = 42) -> None:
        self.d_in = d_in
        self.d_mani = d_mani
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.2, (d_in, d_mani))

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        x: shape (d_in,)
        return shape (d_mani,)
        """
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        h = x @ self.W  # (d_mani,)
        norm = np.linalg.norm(h) + 1e-8
        gamma = np.tanh(norm) / norm
        return gamma * h


# ================================================================
#                  6. Gemini Backend Wrapper
# ================================================================

try:
    from google import genai as google_genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    print("WARNING: google-genai not installed. Run: pip install google-genai")


class GeminiBackend:
    """
    Gemini 2.5 Flash 用ラッパー
    - 埋め込み: text-embedding-004
    - 生成: model_name (デフォルト gemini-2.5-flash)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        if not _GEMINI_AVAILABLE:
            raise RuntimeError("google-genai がインストールされていません。 pip install google-genai")

        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY_HERE":
            raise ValueError("Gemini APIキーが見つかりません。GOOGLE_API_KEY / GEMINI_API_KEY を設定してください。")

        self.client = google_genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.hidden_size = 768  # text-embedding-004 の次元

    # --- Embedding ---

    def get_embedding(self, text: str) -> np.ndarray:
        try:
            resp = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text,
            )
            v = np.array(resp.embeddings[0].values, dtype=float)
            return v
        except Exception as e:
            print(f"[Embedding Error] {e}")
            # フォールバック: テキストから疑似乱数ベクトル
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.normal(0, 1, self.hidden_size)
            v /= np.linalg.norm(v) + 1e-8
            return v

    # --- Text Generation ---

    def generate_one(self, prompt: str, **kwargs: Any) -> str:
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=kwargs or None,
            )
            txt = getattr(resp, "text", None)
            if txt is None:
                # 新しい SDK だと resp.candidates[0].content.parts[0].text の場合がある
                try:
                    cand = resp.candidates[0]
                    parts = cand.content.parts
                    txts = [getattr(p, "text", "") for p in parts]
                    txt = "".join(txts)
                except Exception:
                    txt = ""
            return (txt or "").strip()
        except Exception as e:
            print(f"[Generation Error] {e}")
            return "[Error generating response]"

    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int = 1,
        **kwargs: Any,
    ) -> List[str]:
        out: List[str] = []
        for _ in range(max(1, num_candidates)):
            out.append(self.generate_one(prompt, **kwargs))
        return out


# ================================================================
#                       7. Awakening5X OS
# ================================================================

class Awakening5XOS:
    """
    5軸認知 OS 本体。
    - backend: GeminiBackend
    - d_embed: embedding次元
    - d_mani : manifold 次元（デフォルト 64）
    """

    def __init__(self, backend: GeminiBackend, d_mani: int = 64) -> None:
        self.backend = backend
        self.d_embed = backend.hidden_size
        self.d_mani = d_mani

        self.projector = ManifoldProjector(self.d_embed, d_mani=d_mani)
        self.identity = IdentityStore()
        self.causal = CausalGraph()
        self.essence = EssenceSpace(dim=d_mani)
        self.teleology = Teleology(dim=d_mani)

        self.step: int = 0
        self.last_entity_eid: Optional[int] = None

        # 会話因果ログ（/causal で可視化用）
        self.dialog_causal_log: List[Dict[str, Any]] = []
        self._last_dialog_label: Optional[str] = None

    # ------------------------------------------------------------
    # 基本ユーティリティ
    # ------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        t = text.replace("\n", " ").replace("　", " ")
        raw = t.split()
        return [tok for tok in raw if tok.strip()]

    def _embed_to_mani(self, text: str) -> np.ndarray:
        emb = self.backend.get_embedding(text)
        mani = self.projector.project(emb)
        return mani

    # ------------------------------------------------------------
    #  観測 / Identity / Causal 更新
    # ------------------------------------------------------------

    def observe_text(self, text: str, role: str = "user") -> None:
        """
        テキストを読み取り、Identity / Causal / Essence を更新する。
        """
        self.step += 1
        tokens = self._tokenize(text)
        if not tokens:
            return

        mani = self._embed_to_mani(text)

        self.identity.step_decay()
        prev_eid: Optional[int] = None

        for tok in tokens:
            node = self.identity.get_or_create(tok, mani)
            # Essence にも更新を流す
            self.essence.update(mani)

            if prev_eid is not None:
                self.causal.register(prev_eid, node.eid, weight=1.0)
            prev_eid = node.eid

        # 直前の発話からの継承エッジ
        if self.last_entity_eid is not None and prev_eid is not None:
            self.causal.register(
                self.last_entity_eid,
                prev_eid,
                weight=0.4,
                label=self._last_dialog_label,
            )

        self.last_entity_eid = prev_eid
        self.causal.decay()

    # ------------------------------------------------------------
    #  Teleology ゴール
    # ------------------------------------------------------------

    def add_goal_from_text(
        self,
        name: str,
        description: str,
        exemplar_text: str,
        weight: float = 1.0,
    ) -> None:
        mani = self._embed_to_mani(exemplar_text)
        self.teleology.add_goal(name=name, description=description, vec=mani, weight=weight)

    # ------------------------------------------------------------
    #  Manifold introspection (/mani 用)
    # ------------------------------------------------------------

    def compute_manifold_for_text(self, text: str) -> Dict[str, Any]:
        mani = self._embed_to_mani(text)
        dim = mani.shape[0]
        norm = float(np.linalg.norm(mani))
        max_idx = int(np.argmax(mani))
        min_idx = int(np.argmin(mani))
        max_val = float(mani[max_idx])
        min_val = float(mani[min_idx])

        # Essence 分布
        sims = self.essence.archetypes @ (mani / (norm + 1e-8))
        essence_dist = softmax(sims)

        # Teleology 整合度
        tel_scores = self.teleology.alignment(mani)

        return {
            "text": text,
            "dim": dim,
            "norm": norm,
            "vector": mani,
            "max_idx": max_idx,
            "max_val": max_val,
            "min_idx": min_idx,
            "min_val": min_val,
            "essence_raw": sims,
            "essence_dist": essence_dist,
            "teleology": tel_scores,
        }

    def manifold_report(self, text: str) -> Dict[str, Any]:
        """
        CLI の print_manifold_report() と整合する dict を返す。
        """
        info = self.compute_manifold_for_text(text)
        return {
            "source_text": info["text"],
            "dim": info["dim"],
            "norm": info["norm"],
            "vector": info["vector"],
            "max_idx": info["max_idx"],
            "max_val": info["max_val"],
            "min_idx": info["min_idx"],
            "min_val": info["min_val"],
            "essence_distribution": info["essence_dist"],
            "teleology_alignment": info["teleology"],
        }

    # ------------------------------------------------------------
    #  Teleology に基づく返答生成
    # ------------------------------------------------------------

    def _build_os_prefix(self) -> str:
        """
        Gemini に渡す「OSメタ情報」の前置きテキスト。
        """
        goals_desc = []
        for g in self.teleology.goals:
            goals_desc.append(
                f"- {g['name']}: {g['description']} (weight={g['weight']:.2f})"
            )
        goals_block = "\n".join(goals_desc) if goals_desc else "(no goals)"

        prefix = (
            "You are Awakening5X OS, a 5-axis cognitive shell around a Gemini model.\n"
            "Your behavior must be guided by the following Teleology goals:\n"
            f"{goals_block}\n\n"
            "You MUST:\n"
            "- answer accurately and rigorously (Rigor),\n"
            "- reveal internal structure and reasoning when suitable (Introspect),\n"
            "- behave as an honest partner (Partner) but never dilute rigor.\n"
        )
        return prefix

    def _build_history_block(self, history: List[Tuple[str, str]]) -> str:
        """
        history: [("user" or "assistant", text), ...]
        """
        lines: List[str] = []
        for role, text in history:
            if role == "user":
                lines.append(f"user: {text}")
            else:
                lines.append(f"assistant: {text}")
        return "\n".join(lines)

    def generate_guided_reply(
        self,
        history: List[Tuple[str, str]],
        user_text: str,
        num_candidates: int = 3,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Teleology で rerank された応答を返すメイン入口。

        戻り値は CLI が期待する形に合わせて:
          - reply / best_reply
          - score / teleology_score / best_teleology_score
          - detail / best_teleology_detail
        を含む。
        """

        # --- ユーザー発話を観測 ---
        self.observe_text(user_text, role="user")

        os_prefix = self._build_os_prefix()
        hist_block = self._build_history_block(history)
        prompt = (
            os_prefix
            + "\n"
            + hist_block
            + ("\n" if hist_block else "")
            + f"\nuser: {user_text}\nassistant:"
        )

        # --- 候補生成 ---
        cands = self.backend.generate_candidates(
            prompt,
            num_candidates=num_candidates,
            temperature=temperature,
            top_p=top_p,
        )

        scored: List[Tuple[str, float, Dict[str, float]]] = []

        for cand in cands:
            mani = self._embed_to_mani(cand)
            tel_scores = self.teleology.alignment(mani)
            total = float(sum(tel_scores.values())) if tel_scores else 0.0
            scored.append((cand, total, tel_scores))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_text, best_total, best_tel_scores = scored[0]

        # --- AI 応答も観測に流す ---
        self.observe_text(best_text, role="assistant")

        # best goal
        best_goal_name: Optional[str] = None
        best_goal_score: float = 0.0
        if best_tel_scores:
            best_goal_name = max(best_tel_scores, key=lambda k: best_tel_scores[k])
            best_goal_score = float(best_tel_scores[best_goal_name])

        # 直近の対話ラベルとして保存（次ターンの cross-turn edge に乗る）
        label_for_edge = best_goal_name or "answer"
        self._last_dialog_label = label_for_edge

        # 会話因果ログに追加
        self.dialog_causal_log.append(
            {
                "turn_id": self.step,
                "label": label_for_edge,
                "cosine": float(best_total),
                "teleology_alignment": {k: float(v) for k, v in best_tel_scores.items()},
                "user_text": user_text,
                "reply_text": best_text,
            }
        )

        # 互換性高めた result dict
        result: Dict[str, Any] = {
            # メイン
            "reply": best_text,
            "best_reply": best_text,

            # Teleology 情報
            "score": float(best_total),
            "detail": {k: float(v) for k, v in best_tel_scores.items()},

            # 別名（旧CLI互換用）
            "teleology_score": float(best_total),
            "best_teleology_score": float(best_total),
            "best_teleology_detail": {
                "best_goal": best_goal_name,
                "best_goal_score": float(best_goal_score),
                "scores_by_goal": {k: float(v) for k, v in best_tel_scores.items()},
            },

            # 全候補情報も残しておく
            "candidates": [
                {
                    "text": txt,
                    "score": float(tot),
                    "teleology": {k: float(v) for k, v in tel.items()},
                }
                for (txt, tot, tel) in scored
            ],
        }
        return result

    # ============================================================
    #  Snapshot APIs for CLI (/id, /causal)
    # ============================================================

    def identity_snapshot(self, top_k: int = 32) -> list:
        """
        IdentityStore から上位エンティティを取り出して
        CLI がそのまま表示できる形にする。
        戻り値の各要素は:
          { "eid", "type", "activation", "name" }
        """
        if hasattr(self.identity, "top_entities"):
            ents = self.identity.top_entities(top_k=top_k)
            # すでに dict のリストならそのまま返す
            if ents and isinstance(ents[0], dict):
                return ents

        out = []
        ents = getattr(self.identity, "entities", {})
        for eid, e in ents.items():
            out.append(
                {
                    "eid": int(getattr(e, "eid", eid)),
                    "type": getattr(e, "type_tag", getattr(e, "type", "unknown")),
                    "activation": float(getattr(e, "activation", 0.0)),
                    "name": getattr(e, "name", ""),
                }
            )
        out.sort(key=lambda r: r["activation"], reverse=True)
        return out[:top_k]

    def causal_snapshot(self, top_k: int = 16) -> dict:
        """
        CausalGraph から上位エッジと、会話因果ログを取得して
        CLI の /causal 表示用にまとめる。
        戻り値:
          {
            "edges": [
              {"src", "tgt", "weight", "label", "count"},
              ...
            ],
            "dialog_log": [...]
          }
        """
        edges_list = []
        for src, tgt, w, label, count in self.causal.top_edges(k=top_k):
            edges_list.append(
                {
                    "src": int(src),
                    "tgt": int(tgt),
                    "weight": float(w),
                    "label": label,
                    "count": int(count),
                }
            )

        dialog_log = self.dialog_causal_log
        if isinstance(dialog_log, list) and len(dialog_log) > top_k:
            dialog_log = dialog_log[-top_k:]

        return {
            "edges": edges_list,
            "dialog_log": dialog_log,
        }

    # ------------------------------------------------------------
    #  Causal Graph の簡易ダンプ（おまけ）
    # ------------------------------------------------------------

    def dump_causal_summary(self, top_k: int = 16) -> str:
        """
        強い因果エッジ上位だけ、名前付きでざっくり表示する。
        """
        edges = self.causal.top_edges(k=top_k)
        if not edges:
            return "[Causal] No edges yet."

        lines = ["[Causal Graph Top Edges]"]
        for src, tgt, w, label, count in edges:
            s_node = self.identity.entities.get(str(src))
            t_node = self.identity.entities.get(str(tgt))
            s_name = s_node.name if s_node else f"#{src}"
            t_name = t_node.name if t_node else f"#{tgt}"
            lines.append(
                f"  {src:03d}({s_name}) --> {tgt:03d}({t_name})  "
                f"w={w:0.3f}  label={label}  count={count}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------
    #  Persistence (魂ファイル)
    # ------------------------------------------------------------

    def save_state(self, path: str = "my_soul.json") -> None:
        """
        OS の内部状態を JSON に保存。
        v2.0 フォーマット。
        """
        data: Dict[str, Any] = {}

        data["version"] = "Awakening5X-OS-v2.0"
        data["step"] = self.step

        # Identity
        entities_dump: Dict[str, Any] = {}
        for k, e in self.identity.entities.items():
            entities_dump[k] = {
                "eid": e.eid,
                "name": e.name,
                "type": e.type_tag.name,
                "vec": e.vector.tolist(),
                "activation": e.activation,
                "tags": e.tags,
            }

        data["identity"] = {
            "entities": entities_dump,
            "name_to_id": self.identity.name_to_id,
            "next_eid": self.identity.next_eid,
        }

        # Causal
        data["causal"] = {
            "edges": self.causal.edges,
        }

        # Essence
        data["essence"] = {
            "dim": self.essence.dim,
            "num_archetypes": self.essence.num_archetypes,
            "archetypes": self.essence.archetypes.tolist(),
        }

        # Teleology
        goals_dump = []
        for g in self.teleology.goals:
            goals_dump.append(
                {
                    "name": g["name"],
                    "description": g["description"],
                    "vec": g["vec"].tolist(),
                    "weight": g["weight"],
                }
            )
        data["teleology"] = {
            "dim": self.teleology.dim,
            "goals": goals_dump,
        }

        # Dialog causal log（軽くそのまま持つ）
        data["dialog_causal_log"] = self.dialog_causal_log

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load_state(self, path: str = "my_soul.json") -> None:
        """
        魂ファイルを読み込む。
        - v2.0 フォーマット
        - v1.x の古いフォーマット (entities: {eid: {...}} で eid キーがない形)
          にもある程度互換
        """
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # step
        self.step = int(data.get("step", 0))

        # --- Identity ---
        id_block = data.get("identity")
        if id_block is None:
            # 旧フォーマットへの簡易対応
            raw_entities = data.get("entities", {})
            self.identity = IdentityStore()
            for k, v in raw_entities.items():
                eid = v.get("eid", int(k))
                name = v.get("name", f"E{eid}")
                type_name = v.get("type", v.get("type_tag", "UNKNOWN"))
                try:
                    etype = EntityType[type_name]
                except Exception:
                    etype = EntityType.UNKNOWN
                vec = np.array(v.get("vec", []), dtype=float)
                act = float(v.get("act", v.get("activation", 1.0)))
                node = EntityNode(eid=eid, name=name, type_tag=etype, vector=vec, activation=act)
                self.identity.entities[str(eid)] = node
                self.identity.name_to_id[name.lower()] = eid
                self.identity.next_eid = max(self.identity.next_eid, eid + 1)
        else:
            # v2.0 フォーマット
            self.identity = IdentityStore()
            ent_block = id_block.get("entities", {})
            for k, v in ent_block.items():
                eid = int(v.get("eid", k))
                name = v.get("name", f"E{eid}")
                type_name = v.get("type", "UNKNOWN")
                try:
                    etype = EntityType[type_name]
                except Exception:
                    etype = EntityType.UNKNOWN
                vec = np.array(v.get("vec", []), dtype=float)
                act = float(v.get("activation", 1.0))
                tags = v.get("tags", [])
                node = EntityNode(eid=eid, name=name, type_tag=etype, vector=vec, activation=act, tags=tags)
                self.identity.entities[str(eid)] = node

            # name_to_id / next_eid
            name_to_id = id_block.get("name_to_id", {})
            if not name_to_id:
                for k, e in self.identity.entities.items():
                    self.identity.name_to_id[e.name.lower()] = e.eid
            else:
                self.identity.name_to_id = {k: int(v) for k, v in name_to_id.items()}

            self.identity.next_eid = int(id_block.get("next_eid", len(self.identity.entities)))

        # --- Causal ---
        causal_block = data.get("causal")
        if causal_block is None:
            # 旧フォーマット互換
            self.causal = CausalGraph(edges=data.get("edges", {}))
        else:
            self.causal = CausalGraph(edges=causal_block.get("edges", {}))

        # --- Essence ---
        ess_block = data.get("essence")
        if ess_block is not None:
            dim = int(ess_block.get("dim", self.d_mani))
            num_arch = int(ess_block.get("num_archetypes", 8))
            self.essence = EssenceSpace(dim=dim, num_archetypes=num_arch)
            arr = np.array(ess_block.get("archetypes", []), dtype=float)
            if arr.size > 0:
                self.essence.archetypes = arr
        else:
            self.essence = EssenceSpace(dim=self.d_mani)

        # --- Teleology ---
        tel_block = data.get("teleology")
        if tel_block is not None:
            dim = int(tel_block.get("dim", self.d_mani))
            self.teleology = Teleology(dim=dim)
            for g in tel_block.get("goals", []):
                vec = np.array(g.get("vec", []), dtype=float)
                self.teleology.add_goal(
                    name=g.get("name", "Unknown"),
                    description=g.get("description", ""),
                    vec=vec,
                    weight=float(g.get("weight", 1.0)),
                )
        else:
            # 旧フォーマットに goals が直で入っている場合の簡易対応
            self.teleology = Teleology(dim=self.d_mani)
            for g in data.get("goals", []):
                vec = np.array(g.get("vec", []), dtype=float)
                self.teleology.add_goal(
                    name=g.get("name", "Unknown"),
                    description=g.get("desc", ""),
                    vec=vec,
                    weight=float(g.get("w", 1.0)),
                )

        # --- Dialog causal log ---
        self.dialog_causal_log = data.get("dialog_causal_log", [])
        self._last_dialog_label = None
