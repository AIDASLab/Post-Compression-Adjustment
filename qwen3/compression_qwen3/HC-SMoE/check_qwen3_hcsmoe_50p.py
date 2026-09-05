from hcsmoe.models.qwen.check_compressed import REPO_DIR, run_check


if __name__ == "__main__":
    run_check(
        model_dir=str(REPO_DIR / "qwen3-30b-a3b-hcsmoe-50p-64e"),
        expected_physical=64,
        ratio_label="50%",
    )
