#!/usr/bin/env python3
# Copyright (c) 2026, FAR.AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused tests for root fork identity and PR-to-target identity stability."""

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "fork_base.py"
SPEC = importlib.util.spec_from_file_location("fork_base", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fork_base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fork_base)


class TestForkBaseIdentity(unittest.TestCase):
    def test_root_fork_must_match_origin(self) -> None:
        manifest = {"fork_repo": "https://github.com/AlignmentResearch/Megatron-LM"}
        with patch.object(
            fork_base,
            "_git",
            return_value="https://github.com/example/other-fork.git",
        ):
            self.assertEqual(
                fork_base.check_root_identity(Path("/repo"), manifest),
                [
                    "fork_repo https://github.com/AlignmentResearch/Megatron-LM "
                    "!= origin https://github.com/example/other-fork.git"
                ],
            )

    def test_pull_request_cannot_change_manifest_identity(self) -> None:
        previous = {
            "fork_repo": "https://github.com/AlignmentResearch/Megatron-LM",
            "upstream_repo": "https://github.com/NVIDIA/Megatron-LM",
            "upstream_branch": "main",
            "upstream_base": "a" * 40,
        }
        current = {**previous, "upstream_repo": "https://github.com/example/fork"}
        with patch.object(fork_base, "_git", return_value=json.dumps(previous)):
            problems, notes = fork_base.check_forward(
                Path("/repo"), current, previous["upstream_base"], "origin/farai/main"
            )

        self.assertEqual(notes, [])
        self.assertEqual(
            problems,
            [
                "manifest identity changed for upstream_repo: origin/farai/main "
                "records https://github.com/NVIDIA/Megatron-LM, current records "
                "https://github.com/example/fork"
            ],
        )

    def test_missing_target_manifest_is_a_bootstrap_note(self) -> None:
        missing = subprocess.CalledProcessError(128, ["git", "show"])
        with patch.object(fork_base, "_git", side_effect=missing):
            problems, notes = fork_base.check_forward(
                Path("/repo"), {}, "a" * 40, "origin/farai/main"
            )

        self.assertEqual(problems, [])
        self.assertEqual(
            notes,
            [
                "origin/farai/main has no .fork-base.json yet — "
                "forward-only check not applicable"
            ],
        )


if __name__ == "__main__":
    unittest.main()
