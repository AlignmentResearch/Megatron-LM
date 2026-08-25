# FAR.AI Fork of NVIDIA/Megatron-LM

Fork of [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), tracking upstream `main`.

Everything fork-specific is on this page. Upstream documentation ([README.md](README.md),
[CONTRIBUTING.md](CONTRIBUTING.md), [docs/](docs/)) applies unchanged.


## Working in this fork

`farai/main` is the default branch. Branch off it, and land every PR with **"Create a merge commit"** — including
upstream syncs. Force-pushing `farai/main` is not allowed.

```sh
make help    # all commands
```


## FAR.AI Patches

| # | Change | Files |
|---|---|---|
| 1 | [Pruned NVIDIA-only GitHub workflows](#1-pruned-nvidia-only-github-workflows) | `.github/` |
| 2 | [Fork base manifest](#2-fork-base-manifest) | `.fork-base.json`, `tools/fork_base.py`, `tools/sync_upstream.sh`, `.github/workflows/fork-base.yml` |
| 3 | [Makefile](#3-makefile) | `Makefile` |
| 4 | Expert-LoRA GEMM stack and hybrid-recompute guard — pending in [PR #1](https://github.com/AlignmentResearch/Megatron-LM/pull/1) (branch `tf-at/moe-lora-stack-on-d12f6c8c`) | `megatron/core/` |

### 1. Pruned NVIDIA-only GitHub workflows

Upstream's GitHub workflows are NVIDIA-org automation that cannot run here: every one is either gated on
`github.repository == 'NVIDIA/Megatron-LM'` or needs infrastructure this fork does not have — NVIDIA's
`FW-CI-templates`, the `copy-pr-bot` app, self-hosted runners, and a list of org secrets (`PAT`,
`NVIDIA_INFERENCE_*`, `TWINE_PASSWORD`, Slack webhooks). All of `.github/workflows/` was removed, along with
`CODEOWNERS` (it names `@NVIDIA/*` teams that do not exist here) and `copy-pr-bot.yaml`.

Megatron-LM's real test CI is GitLab (`.gitlab-ci.yml`, `.gitlab/`). GitLab CI does not run on GitHub, so those
files are inert here and were left in place — deleting them would only add conflict surface on every upstream
sync. The same goes for the workflow helpers under `.github/actions/` and `.github/scripts/`.

The unit-test suite is GPU-bound and is not brought up on fork CI.

### 2. Fork base manifest

`.fork-base.json` records the upstream commit this fork is based on. NeMo-RL pins this repo as a submodule and
reads that record to tell whether a commit here is compatible with a commit there.

```sh
make fork-base          # regenerate after a sync merge, then commit .fork-base.json
make fork-base-check    # verify it against git (what CI runs)
make fork-base-print    # print the base commit
```

`.github/workflows/fork-base.yml` runs the check on every PR and on pushes to `farai/main`, publishing the status
context `fork-base`. `tools/fork_base.py` is shared verbatim with the other FAR.AI forks — do not edit it here
alone.

### 3. Makefile

`make help` lists the commands. Upstream ships no Makefile, so this one is fork-only.


## Keeping the Fork in Sync with Upstream

Add the upstream remote once:

```sh
git remote add upstream https://github.com/NVIDIA/Megatron-LM.git
```

Then:

```sh
make sync-upstream                                  # merges upstream, updates the manifest
git push -u origin sync/upstream-<date>-<sha>
```

Open the branch as a PR against `farai/main` and land it with **"Create a merge commit"**. Squashing a sync PR
leaves upstream's commits out of our history and breaks the recorded base.

When resolving conflicts, don't take `--ours` wholesale — that drops upstream's changes to a file while the
recorded base still claims we contain them.
