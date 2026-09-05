from hcsmoe.models.qwen.check_compressed import REPO_DIR, run_check


if __name__ == "__main__":
    run_check(
        model_dir=str(REPO_DIR / "qwen3-30b-a3b-hcsmoe-625p-80e"),
        expected_physical=80,
        ratio_label="62.5%",
    )
