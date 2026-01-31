import click
from pathlib import Path

from main import DatasetQualityAuditor
from utils.helpers import create_sample_data, ensure_dir


@click.group()
def cli():
    """Dataset Quality Auditor - Assess ML readiness of datasets"""
    pass


@cli.command()
@click.option(
    "--file",
    "-f",
    required=True,
    type=click.Path(exists=True),
    help="Path to dataset file or directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output directory for reports",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="config/config.yaml",
    help="Configuration file path",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def audit(file, output, config, verbose):
    """Run quality audit on a dataset"""
    click.echo(click.style("\n🔍 Dataset Quality Auditor", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))

    try:
        auditor = DatasetQualityAuditor(config_path=config)

        with click.progressbar(length=6, label="Running audit") as bar:
            results = auditor.audit_dataset(file_path=file, output_dir=output)
            bar.update(6)

        click.echo(click.style("\n✅ Audit Complete!", fg="green", bold=True))
        click.echo(
            f"\nOverall Score: "
            f"{click.style(str(results['scores']['overall_score']), fg='yellow', bold=True)}/100"
        )
        click.echo(
            f"Rating: {click.style(results['scores']['rating'], fg='green')}"
        )
        click.echo(
            f"\n📁 Reports saved to: "
            f"{click.style(results['output_dir'], fg='blue')}"
        )

    except Exception as e:
        click.echo(click.style(f"\n❌ Error: {str(e)}", fg="red"))
        raise click.Abort()


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/sample_tabular/sample_dataset.csv",
    help="Output path for sample dataset",
)
@click.option(
    "--samples",
    "-n",
    type=int,
    default=1000,
    help="Number of samples to generate",
)
@click.option(
    "--features",
    "-f",
    type=int,
    default=10,
    help="Number of features to generate",
)
@click.option(
    "--anomaly-ratio",
    "-a",
    type=float,
    default=0.1,
    help="Ratio of anomalies (0.0-1.0)",
)
def generate_sample(output, samples, features, anomaly_ratio):
    """Generate sample dataset for testing"""
    click.echo(click.style("\n📊 Generating Sample Dataset", fg="cyan", bold=True))

    try:
        df = create_sample_data(
            n_samples=samples,
            n_features=features,
            anomaly_ratio=anomaly_ratio,
        )

        output_path = Path(output)
        ensure_dir(output_path.parent)
        df.to_csv(output_path, index=False)

        click.echo(click.style("\n✅ Sample dataset created!", fg="green"))
        click.echo(f"Path: {click.style(str(output_path), fg='blue')}")
        click.echo(f"Samples: {samples}, Features: {features}")

    except Exception as e:
        click.echo(click.style(f"\n❌ Error: {str(e)}", fg="red"))
        raise click.Abort()


@cli.command()
def info():
    """Display system information"""
    import sys
    import platform
    import numpy as np
    import pandas as pd
    import sklearn
    import tensorflow as tf

    click.echo(click.style("\n📋 System Information", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))

    click.echo(f"\nPython: {sys.version.split()[0]}")
    click.echo(f"Platform: {platform.platform()}")
    click.echo("\nLibrary Versions:")
    click.echo(f"  • NumPy: {np.__version__}")
    click.echo(f"  • Pandas: {pd.__version__}")
    click.echo(f"  • Scikit-learn: {sklearn.__version__}")
    click.echo(f"  • TensorFlow: {tf.__version__}")
    click.echo()


if __name__ == "__main__":
    cli()
