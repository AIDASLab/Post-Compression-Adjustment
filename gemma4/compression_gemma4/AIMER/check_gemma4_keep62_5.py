from pathlib import Path

from src.gemma4_sanity_check import main


if __name__ == "__main__":
    main(
        default_pruned_dir=str(Path(__file__).resolve().parent / "gemma4_aimer_keep62_5"),
        default_keep_experts=80,
    )
