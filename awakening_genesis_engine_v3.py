# -*- coding: utf-8 -*-
"""
awakening_genesis_engine_v3.py
------------------------------
Awakening Genesis Dynamics Engine v3.0

- OS（Awakening5XOS_V3）の「概念多様体ベクトル」を受け取り、
  低次元のダイナミクス空間に写して流れ（trajectory）を計算する。
- CLI の /flow コマンドから呼ばれる。

公開API:
  - AwakeningGenesisEngine(dim=4)
      * attach(os_core)
      * run_flow(mani_vec)

設計方針:
  - OS が「高次元の概念多様体」を持ち、
    Genesis はそれを 4次元くらいの「ダイナミクス空間」に圧縮して
    時間発展を眺める役割。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np


@dataclass
class AwakeningGenesisEngine:
    """
    dim: ダイナミクス空間の次元（デフォルト4）
    steps: 何ステップ流すか
    step_size: 更新の強さ
    """

    dim: int = 4
    steps: int = 32
    step_size: float = 0.12

    # OS との接続
    os_core: Optional[Any] = None
    projector: Optional[Any] = None  # project(mani_vec) -> z0 in R^dim

    def attach(self, os_core: Any) -> None:
        """
        Awakening5XOS_V3 を受け取り、
        そこからダイナミクス用の単純な射影器を構成する。
        """
        self.os_core = os_core

        # OS 側は高次元 manifold ベクトルを返すので、
        # ここでは「先頭 dim 次元だけを抜き出す射影器」を自前で定義する。
        class _SimpleProjector:
            def __init__(self, dim: int) -> None:
                self.dim = dim

            def project(self, v: np.ndarray) -> np.ndarray:
                arr = np.array(v, dtype=float).ravel()
                if arr.size < self.dim:
                    pad = np.zeros(self.dim - arr.size, dtype=float)
                    arr = np.concatenate([arr, pad])
                return arr[: self.dim]

        self.projector = _SimpleProjector(self.dim)

    # ------------------------------------------------------------
    # 内部ダイナミクス
    # ------------------------------------------------------------

    def _flow_step(self, z: np.ndarray) -> np.ndarray:
        """
        シンプルな非線形ダイナミクス。
        ここでは「中心に向かう減衰 + tanh 非線形」を入れているだけ。

        z_{t+1} = z_t + step_size * ( -z_t + tanh(2 z_t) )
        """

        z = np.array(z, dtype=float)
        nonlin = np.tanh(2.0 * z)
        dz = -z + nonlin
        return z + self.step_size * dz

    # ------------------------------------------------------------
    # 公開API: /flow 用
    # ------------------------------------------------------------

    def run_flow(self, mani_vec: np.ndarray) -> Dict[str, Any]:
        """
        OS の manifold ベクトル (高次元) を受け取り、
        低次元 z 空間で trajectory を生成する。

        Parameters
        ----------
        mani_vec : np.ndarray
            Awakening5XOS_V3.compute_manifold_for_text(...) などから
            取得した "vector" を想定。

        Returns
        -------
        dict:
          {
            "initial": [float, ...],   # 初期状態 z0
            "trajectory": [[...], ...],# 各ステップの z_t
            "steps": int               # ステップ数
          }
        """
        if self.projector is None:
            raise RuntimeError(
                "Genesis engine is not attached to OS. "
                "Call `genesis.attach(os_core)` before run_flow()."
            )

        # 1. manifold ベクトルをダイナミクス空間へ射影
        z0 = self.projector.project(mani_vec)
        z = np.array(z0, dtype=float)

        traj: List[np.ndarray] = [z.copy()]
        for _ in range(self.steps):
            z = self._flow_step(z)
            traj.append(z.copy())

        return {
            "initial": z0.tolist(),
            "trajectory": [t.tolist() for t in traj],
            "steps": len(traj),
        }
