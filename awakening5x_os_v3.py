# -*- coding: utf-8 -*-
"""
Awakening5X OS v3.0
--------------------
Core 5-axis cognitive OS for stateless LLM backends (Gemini / GPT / Claude).

Axes:
  1. Manifold Projection (semantic geometry)
  2. Identity Store (persistent entities)
  3. Causal Graph (v3 stabilized)
  4. Essence Space (archetype attractors)
  5. Teleology Field (goal gravity for reranking)

This file contains ONLY the OS.
Genesis Dynamics Engine is in: awakening_genesis_engine_v3.py
CLI is in: awakening_genesis_cli_v3.py
"""

from __future__ import annotations
import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto

# ============================================================
# Utility
# ============================================================

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = x.astype(float)
    x = x - np.max(x)
    ex = np.exp(x / max(temp, 1e-8))
    s = np.sum(ex)
    return ex / (s + 1e-8)

# ============================================================
# Entity System (Identity Store)
# ============================================================

class EntityType(Enum):
    UNKNOWN = auto()
    PERSON = auto()
    OBJECT = auto()
    CONCEPT = auto()
    EVENT = auto()

def guess_entity_type(token: str) -> EntityType:
    if token.istitle():
        return EntityType.PERSON
    if token.endswith("論") or token.endswith("性"):
        return EntityType.CONCEPT
    return EntityType.UNKNOWN

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
    entities: Dict[str, EntityNode] = field(default_factory=dict)
    name_to_id: Dict[str, int] = field(default_factory=dict)
    next_eid: int = 0

    def step_decay(self, rate: float = 0.97):
        for e in self.entities.values():
            e.activation *= rate

    def get_or_create(self, name: str, vec: np.ndarray) -> EntityNode:
        key = name.lower().strip()
        if not key:
            key = "<EMPTY>"

        if key in self.name_to_id:
            eid = self.name_to_id[key]
            node = self.entities[str(eid)]
            node.vector = 0.85 * node.vector + 0.15 * vec
            node.activation = 1.0
            return node

        eid = self.next_eid
        self.next_eid += 1
        etype = guess_entity_type(name)
        node = EntityNode(eid=eid, name=name, type_tag=etype, vector=vec)
        self.entities[str(eid)] = node
        self.name_to_id[key] = eid
        return node

    def top_entities(self, top_k=20):
        ents = sorted(self.entities.values(), key=lambda e: e.activation, reverse=True)
        return [
            {
                "eid": e.eid,
                "name": e.name,
                "type": e.type_tag.name,
                "activation": float(e.activation),
            }
            for e in ents[:top_k]
        ]

# ============================================================
# Causal Graph v3
# ============================================================

@dataclass
class CausalGraph:
    edges: Dict[str, dict] = field(default_factory=dict)
    decay_rate: float = 0.985

    def _ensure(self, src: int, tgt: int):
        key = f"{src}->{tgt}"
        rec = self.edges.get(key)
        if isinstance(rec, dict):
            return key, rec
        # convert float → dict
        w = float(rec) if rec else 0.0
        rec = {"weight": w, "count": 0, "label": None}
        self.edges[key] = rec
        return key, rec

    def register(self, src: int, tgt: int, weight: float, label=None):
        if src is None or tgt is None or src == tgt:
            return
        key, rec = self._ensure(src, tgt)
        rec["weight"] = 0.92 * rec["weight"] + 0.08 * weight
        rec["count"] += 1
        if label:
            rec["label"] = label

    def decay(self):
        dead = []
        for k, rec in list(self.edges.items()):
            rec["weight"] *= self.decay_rate
            if abs(rec["weight"]) < 0.005:
                dead.append(k)
        for k in dead:
            del self.edges[k]

    def top_edges(self, k=16):
        items = []
        for key, rec in self.edges.items():
            src, tgt = key.split("->")
            items.append((
                int(src), int(tgt),
                float(rec["weight"]),
                rec.get("label"),
                int(rec.get("count", 0))
            ))
        items.sort(key=lambda x: abs(x[2]), reverse=True)
        return items[:k]

# ============================================================
# Essence Space
# ============================================================

@dataclass
class EssenceSpace:
    dim: int
    num_archetypes: int = 8
    lr: float = 0.07

    def __post_init__(self):
        rng = np.random.default_rng(1234)
        raw = rng.normal(0, 1, (self.num_archetypes, self.dim))
        self.archetypes = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)

    def update(self, v: np.ndarray):
        v = v / (np.linalg.norm(v) + 1e-8)
        sims = self.archetypes @ v
        idx = int(np.argmax(sims))
        self.archetypes[idx] += self.lr * (v - self.archetypes[idx])
        self.archetypes[idx] /= np.linalg.norm(self.archetypes[idx]) + 1e-8
        return sims

# ============================================================
# Teleology v3 - Goal Gravity Field
# ============================================================

@dataclass
class Teleology:
    dim: int
    goals: List[Dict[str, Any]] = field(default_factory=list)

    def add_goal(self, name: str, desc: str, vec: np.ndarray, weight=1.0):
        v = vec / (np.linalg.norm(vec) + 1e-8)
        self.goals.append({
            "name": name,
            "description": desc,
            "vec": v,
            "weight": float(weight),
        })

    def alignment(self, vec: np.ndarray) -> Dict[str, float]:
        if not self.goals:
            return {}
        v = vec / (np.linalg.norm(vec) + 1e-8)
        scores = {}
        for g in self.goals:
            sim = float(np.dot(v, g["vec"]))
            # v3: “gravity” model
            scores[g["name"]] = g["weight"] * sim
        return scores

# ============================================================
# Manifold Projector v3
# ============================================================

class ManifoldProjector:
    def __init__(self, d_in, d_mani=64, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.15, (d_in, d_mani))

    def project(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        h = x @ self.W
        norm = np.linalg.norm(h) + 1e-8
        gamma = np.tanh(norm) / norm
        return gamma * h

# ============================================================
# Awakening5X OS (v3)
# ============================================================
class Awakening5XOS_V3:
    def __init__(self, backend=None, d_mani: int = 64):
        """
        backend:
            None の場合は、自分で GeminiBackend を生成する。
            既存の Backend を渡したい場合は、外から渡してもよい。
        """
        if backend is None:
            try:
                # ★ ここでだけ GeminiBackend を import
                from awakening5x_api_os_gemini import GeminiBackend
            except ImportError as e:
                raise RuntimeError(
                    "GeminiBackend が見つかりません。\n"
                    "awakening5x_api_os_gemini.py が同じフォルダにあるか確認してください。"
                ) from e
            backend = GeminiBackend()

        self.backend = backend
        self.d_embed = backend.hidden_size
        self.d_mani = d_mani

        # 以下は元の初期化ロジックそのまま
        self.projector = ManifoldProjector(self.d_embed, d_mani=d_mani)
        self.identity = IdentityStore()
        self.causal = CausalGraph()
        self.essence = EssenceSpace(dim=d_mani)
        self.teleology = Teleology(dim=d_mani)

        self.step: int = 0
        self.last_entity_eid: Optional[int] = None
        self.dialog_causal_log: list[dict] = []
        self._last_dialog_label: Optional[str] = None
        self.projector = ManifoldProjector(self.d_embed, d_mani)
        self.identity = IdentityStore()
        self.causal = CausalGraph()
        self.essence = EssenceSpace(dim=d_mani)
        self.teleology = Teleology(dim=d_mani)

        self.step = 0
        self.last_entity = None
        self._last_label = None
        self.dialog_log = []

    # ----------------- Tokenize -----------------
    def _tokenize(self, text):
        return [t for t in text.replace("\n", " ").split() if t.strip()]

    def _embed_mani(self, text):
        e = self.backend.get_embedding(text)
        return self.projector.project(e)

    # ---------------- Observe -------------------
    def observe_text(self, text, role="user"):
        self.step += 1
        toks = self._tokenize(text)
        if not toks:
            return

        mani = self._embed_mani(text)
        self.identity.step_decay()

        prev = None
        for tok in toks:
            node = self.identity.get_or_create(tok, mani)
            self.essence.update(mani)

            if prev is not None:
                self.causal.register(prev.eid, node.eid, weight=1.0)
            prev = node

        if self.last_entity and prev:
            self.causal.register(self.last_entity.eid, prev.eid, weight=0.32, label=self._last_label)

        self.last_entity = prev
        self.causal.decay()

    # ---------------- Teleology -----------------
    def add_goal_from_text(self, name, desc, exemplar):
        mani = self._embed_mani(exemplar)
        self.teleology.add_goal(name, desc, mani)
    
    def compute_manifold_for_text(self, text: str) -> dict:
        """
        テキストを Awakening の多様体に埋め込み、
        /flow 用にベクトル情報を返すユーティリティ。
        """
        # 1. すでにある _embed_mani を使って射影
        mani = self._embed_mani(text)

        # 2. numpy ベクトルにしてスカラー情報を計算
        vec = np.asarray(mani, dtype=float)
        dim = int(vec.shape[0])
        norm = float(np.linalg.norm(vec) + 1e-8)
        max_idx = int(np.argmax(vec))
        min_idx = int(np.argmin(vec))
        max_val = float(vec[max_idx])
        min_val = float(vec[min_idx])

        return {
            "text": text,
            "dim": dim,
            "norm": norm,
            "vector": vec.tolist(),  # CLI 側で np.array(...) し直す
            "max_idx": max_idx,
            "max_val": max_val,
            "min_idx": min_idx,
            "min_val": min_val,
        }

    # -------------- Manifold Report -------------
    def manifold_report(self, text):
        mani = self._embed_mani(text)
        norm = float(np.linalg.norm(mani))
        sims = self.essence.archetypes @ (mani / (norm + 1e-8))
        tel = self.teleology.alignment(mani)

        return {
            "text": text,
            "norm": norm,
            "max_idx": int(np.argmax(mani)),
            "min_idx": int(np.argmin(mani)),
            "essence_distribution": softmax(sims),
            "teleology_alignment": tel,
        }

    # ---------------- Reranked Reply -------------
    def generate_guided_reply(self, history, user_text, num_candidates=3):
        self.observe_text(user_text, role="user")

        prefix = "You are Awakening5X OS v3.0.\nYour goals:\n"
        for g in self.teleology.goals:
            prefix += f"- {g['name']} (w={g['weight']})\n"

        htxt = ""
        for r, t in history:
            htxt += f"{r}: {t}\n"

        prompt = prefix + "\n" + htxt + f"user: {user_text}\nassistant:"

        cands = self.backend.generate_candidates(prompt, num_candidates=num_candidates)
        scored = []
        for c in cands:
            mani = self._embed_mani(c)
            tel = self.teleology.alignment(mani)
            total = sum(tel.values()) if tel else 0.0
            scored.append((c, total, tel))

        scored.sort(key=lambda x: x[1], reverse=True)
        best, best_total, best_tel = scored[0]

        self.observe_text(best, role="assistant")

        best_goal = max(best_tel, key=lambda k: best_tel[k]) if best_tel else None
        self._last_label = best_goal

        self.dialog_log.append({
            "turn": self.step,
            "goal": best_goal,
            "score": best_total,
            "user": user_text,
            "reply": best,
        })

        return {
            "reply": best,
            "score": best_total,
            "goals": best_tel,
            "candidates": [
                {"text": c, "score": s, "goals": g}
                for (c, s, g) in scored
            ],
        }

    # ---------------- Identity Snapshot ---------
    def identity_snapshot(self):
        return self.identity.top_entities(30)

    # ---------------- Causal Snapshot -----------
    def causal_snapshot(self):
        return {
            "edges": [
                {
                    "src": s,
                    "tgt": t,
                    "weight": w,
                    "label": lab,
                    "count": c,
                }
                for (s, t, w, lab, c) in self.causal.top_edges(16)
            ],
            "dialog_log": self.dialog_log[-16:],
        }

    # ---------------- Persistence ----------------
    def save_state(self, path="my_soul.json"):
        data = {
            "version": "Awakening5X-OS-v3",
            "step": self.step,
            "identity": {
                "entities": {
                    k: {
                        "eid": e.eid,
                        "name": e.name,
                        "type": e.type_tag.name,
                        "vec": e.vector.tolist(),
                        "activation": e.activation,
                        "tags": e.tags,
                    }
                    for k, e in self.identity.entities.items()
                },
                "name_to_id": self.identity.name_to_id,
                "next_eid": self.identity.next_eid,
            },
            "causal": {"edges": self.causal.edges},
            "essence": {
                "dim": self.essence.dim,
                "num_archetypes": self.essence.num_archetypes,
                "archetypes": self.essence.archetypes.tolist(),
            },
            "teleology": {
                "dim": self.teleology.dim,
                "goals": [
                    {
                        "name": g["name"],
                        "description": g["description"],
                        "vec": g["vec"].tolist(),
                        "weight": g["weight"],
                    }
                    for g in self.teleology.goals
                ],
            },
            "dialog_log": self.dialog_log,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load_state(self, path="my_soul.json"):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.step = data.get("step", 0)

        # identity
        idb = data.get("identity", {})
        ents = idb.get("entities", {})
        self.identity = IdentityStore()
        for k, v in ents.items():
            eid = v.get("eid", int(k))
            name = v.get("name", f"E{eid}")
            typ = v.get("type", "UNKNOWN")
            vec = np.array(v.get("vec", []), float)
            act = float(v.get("activation", 1))
            tags = v.get("tags", [])
            node = EntityNode(eid=eid, name=name, type_tag=EntityType[typ], vector=vec, activation=act, tags=tags)
            self.identity.entities[str(eid)] = node
            self.identity.name_to_id[name.lower()] = eid
        self.identity.next_eid = idb.get("next_eid", len(ents))

        # causal
        self.causal = CausalGraph(edges=data.get("causal", {}).get("edges", {}))

        # essence
        ess = data.get("essence", {})
        self.essence = EssenceSpace(dim=ess.get("dim", self.d_mani), num_archetypes=ess.get("num_archetypes", 8))
        arr = np.array(ess.get("archetypes", []), float)
        if arr.size > 0:
            self.essence.archetypes = arr

        # teleology
        self.teleology = Teleology(dim=data.get("teleology", {}).get("dim", self.d_mani))
        for g in data.get("teleology", {}).get("goals", []):
            vec = np.array(g["vec"], float)
            self.teleology.add_goal(g["name"], g["description"], vec, g["weight"])

        self.dialog_log = data.get("dialog_log", [])
