# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIAGNOSTIC ONLY (wip/sigsched-rubin-diag, never for a PR): environment knobs that take pieces out of the streaming
signal consumer's per-tile path, to attribute its overhead on Rubin with every M tile ready at launch.  Read when a
kernel is built (host Python), so each knob setting is a separate compile.  Unsafe with a live producer.

  SIGDIAG_NO_CLAIM_SPIN=1     skip the per-claim acquire of signal[m_tile] (every tile is ready anyway)
  SIGDIAG_NO_PROXY_FENCE=1    skip the TMA warp's per-tile fence.proxy.async.global
  SIGDIAG_TILE_STAGES=<n>     tile_info pipeline depth (the scheduler warp's run-ahead), default 2
  SIGDIAG_BATCH_CAP=<n>       EXPERT_BATCH_MAX_M_TILES (default 16), read at import: one value per process
"""

import os


def diag(name, default=None):
    return os.environ.get("SIGDIAG_" + name, default)
