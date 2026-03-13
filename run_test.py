from pcb11.pipeline import run_pipeline, PipelineConfig
from pathlib import Path

config = PipelineConfig(
    model="alexnet",
    output_root=Path("outputs/run_alexnet")
)

run_pipeline(config)
