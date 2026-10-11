"""Sequential Notebook Executor for XMCA-Net-Lite.

Executes notebooks 00 to 09 in strict order, saving executed outputs
and logging progress and status for each step.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("NotebookRunner")

NOTEBOOK_SEQUENCE = [
    "00_environment_data.ipynb",
    "01_prepare_corpora.ipynb",
    "02_generate_qwen.ipynb",
    "03_generate_llava.ipynb",
    "04_perception.ipynb",
    "05_retrieval.ipynb",
    "06_utilization.ipynb",
    "07_combiner.ipynb",
    "08_ablation.ipynb",
    "09_reporting.ipynb",
]


def execute_notebook(notebook_path: Path, timeout: int = 1800) -> bool:
    """Execute a single Jupyter notebook using nbclient."""
    import nbformat
    from nbclient import NotebookClient

    logger.info(f"==> Starting execution: {notebook_path.name}")
    start_time = time.time()

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        import os
        os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}"

        effective_timeout = None if (timeout is None or timeout <= 0) else timeout
        client = NotebookClient(
            nb,
            timeout=effective_timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(PROJECT_ROOT)}},
        )

        client.execute()

        # Save executed notebook with outputs in place
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        elapsed = time.time() - start_time
        logger.info(f"✓ Completed {notebook_path.name} in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ Failed {notebook_path.name} after {elapsed:.1f}s: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run XMCA-Net-Lite notebooks in sequence."
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Notebook index to start from (0 to 9, default: 0)",
    )
    parser.add_argument(
        "--stop-at",
        type=int,
        default=9,
        help="Notebook index to stop at (0 to 9, default: 9)",
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default=None,
        help="Run only a specific notebook by name (e.g. 00_environment_data.ipynb)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds per cell (default: None / unlimited)",
    )

    args = parser.parse_args()
    notebooks_dir = PROJECT_ROOT / "notebooks"

    if args.notebook:
        target = notebooks_dir / args.notebook
        if not target.exists():
            logger.error(f"Notebook {target} not found!")
            sys.exit(1)
        success = execute_notebook(target, timeout=args.timeout)
        sys.exit(0 if success else 1)

    selected = NOTEBOOK_SEQUENCE[args.start_from : args.stop_at + 1]
    logger.info(f"Selected {len(selected)} notebooks to run sequentially:")
    for i, name in enumerate(selected, start=args.start_from):
        logger.info(f"  [{i:02d}] {name}")

    total_start = time.time()
    for i, name in enumerate(selected, start=args.start_from):
        nb_path = notebooks_dir / name
        if not nb_path.exists():
            logger.error(f"Notebook file not found: {nb_path}")
            sys.exit(1)

        success = execute_notebook(nb_path, timeout=args.timeout)
        if not success:
            logger.error(
                f"Halting sequence at [{i:02d}] {name} due to execution error."
            )
            sys.exit(1)

    total_elapsed = time.time() - total_start
    logger.info(
        f"All {len(selected)} notebooks executed successfully in {total_elapsed:.1f}s!"
    )


if __name__ == "__main__":
    main()
