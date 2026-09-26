"""Train all HemoTest ML models."""
import sys
import typer, pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from hemotest.data.generator import CalibrationDataGenerator
from hemotest.data.preprocessing import preprocess_features
from hemotest.models.ml_models import HemoTestModelTrainer, FEATURE_COLS

app = typer.Typer()
console = Console()

@app.command()
def train(
    data_dir: Path = typer.Option(Path("data/processed")),
    output_dir: Path = typer.Option(Path("models")),
    generate_data: bool = typer.Option(False, "--generate-data"),
    n_samples: int = typer.Option(500),
    random_state: int = typer.Option(42),
):
    console.rule("[bold blue]HemoTest ML — NeuroVitalis / EQUIDX AI")
    # Typer 0.12 paired with newer Click releases can leave a boolean flag at
    # its default even though Click recognized the option. Keep the documented
    # ``--generate-data`` command reliable across supported environments.
    generate_data = generate_data or "--generate-data" in sys.argv
    if generate_data:
        gen = CalibrationDataGenerator(random_state=random_state)
        df = gen.generate_calibration_dataset(n_samples=n_samples, nigeria_distribution=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_dir / "calibration_data.csv", index=False)
        console.print(f"[green]Generated {len(df)} samples")
    else:
        csvs = list(data_dir.glob("*.csv"))
        if not csvs:
            console.print(f"[red]No CSV in {data_dir}. Use --generate-data"); raise typer.Exit(1)
        df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)

    df = preprocess_features(df)
    trainer = HemoTestModelTrainer(feature_cols=FEATURE_COLS)
    metrics = trainer.train_all(df)

    t = Table(show_lines=True)
    for c in ["Model","RMSE","MAE","R2","CV-RMSE"]:
        t.add_column(c)
    for name, m in metrics.items():
        t.add_row(name, f"{m.rmse:.3f}", f"{m.mae:.3f}", f"{m.r2:.4f}",
                  f"{m.cv_rmse_mean:.3f}+/-{m.cv_rmse_std:.3f}")
    console.print(t)

    trainer.save_all(output_dir)
    trainer.save_best(output_dir / "ensemble_v1.pkl")
    console.print(f"[bold green]Done — models saved to {output_dir}/")

if __name__ == "__main__":
    app()
