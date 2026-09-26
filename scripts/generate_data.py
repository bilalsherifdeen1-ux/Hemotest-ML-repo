"""Generate synthetic calibration dataset."""
import typer, pandas as pd
from pathlib import Path
from hemotest.data.generator import CalibrationDataGenerator

def main(
    n_samples: int = typer.Option(500),
    output: Path = typer.Option(Path("data/processed/calibration_data.csv")),
    random_state: int = typer.Option(42),
):
    gen = CalibrationDataGenerator(random_state=random_state)
    df = gen.generate_calibration_dataset(n_samples=n_samples, nigeria_distribution=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} samples -> {output}")

if __name__ == "__main__":
    typer.run(main)
