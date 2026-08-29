# Megatron-LM (FAR.AI) Makefile. See README.farai.md for the fork overview.

.DEFAULT_GOAL := help

.PHONY: help fork-base fork-base-check fork-base-print sync-upstream

help:
	@echo ""
	@echo "Megatron-LM (FAR.AI) development commands"
	@echo "========================================="
	@echo ""
	@echo "Fork base (which upstream commit this fork is based on; consumed by NeMo-RL):"
	@echo "  make fork-base            Regenerate .fork-base.json — run after a sync, then commit it"
	@echo "  make fork-base-check      Verify the manifest against git (what CI runs)"
	@echo "  make fork-base-print      Print the computed base commit"
	@echo "  make sync-upstream        Merge newer upstream into this fork (creates a sync/ branch)"
	@echo "                            UPSTREAM_REF=<commit> to target a specific commit; DRY_RUN=1 to preview"
	@echo ""

# tools/fork_base.py is shared verbatim with the other FAR.AI forks — do not edit it here alone.
fork-base:
	@python3 tools/fork_base.py --write

fork-base-check:
	@python3 tools/fork_base.py --check

fork-base-print:
	@python3 tools/fork_base.py --print

sync-upstream:
	@bash tools/sync_upstream.sh
