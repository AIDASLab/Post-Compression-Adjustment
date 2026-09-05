import os
from pathlib import Path

from check_gemma4_compressed_model import check_model


if __name__ == "__main__":
    check_model(
        model_dir=os.environ.get(
            "MCSMOE_MODEL_DIR",
            str(Path(__file__).resolve().parent / "gemma4-26b-a4b-it-mcsmoe-50"),
        ),
        expected_physical_num_experts=64,
    )
