# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Scan the tabletop eval config for blocked tokens stored as sha256 digests."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

DENYLIST = {
    "641b26b6c6f8a09ce6ef3dac54768fe25517296ad6a5af7660b2726132d5bea4",
    "fab16132ae80815d3eec988aab6eae703c71db0fd0cae9b2b8e02dae2383c671",
    "2014e7b6016a99da47d622d416fe305d2651f3a7730a6b0cb640344a486808b9",
    "40a3edd1943cfb35de99107ac535dc0b98f716b862da72b3d8ca0ec4da3610b6",
    "af9aadb68121b1c2766c181a808b5c55294854e0065129e2ea639bf5984c5a28",
    "524b35430349cfa253f5ad8fc6622a74f0c19fef62a7d4335452466078f38656",
    "1352fc379f25acc10cd92617e4506564c07dacc7b56e69ca1790635a5c8433e4",
    "e9925cf52a3eaadff9d26585177dd9845fabc278e0536538ca8d18fa08c012b1",
}


def test_eval_config_has_no_blocked_tokens():
    path = Path(__file__).resolve().parents[1] / "configs" / "task-franka-tabletop.yaml"
    text = path.read_text(encoding="utf-8")
    hits = [
        token
        for token in re.findall(r"[A-Za-z0-9_]+", text)
        if hashlib.sha256(token.lower().encode("utf-8")).hexdigest() in DENYLIST
    ]
    assert hits == []
