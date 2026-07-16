"""stemsurf CLI.

Examples:
    stemsurf song.mp3
    stemsurf song.mp3 -o mixes/ --model spleeter:5stems --lufs -12
    stemsurf song.mp3 --stems-dir my_daw_stems/   # skip separation
"""

import click

from .config import MixConfig
from .pipeline import MixPipeline
from .separation import FolderEngine, SpleeterEngine


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="output", show_default=True,
              help="Where to write the remix and stems.")
@click.option("--model", default="spleeter:4stems", show_default=True,
              type=click.Choice(["spleeter:2stems", "spleeter:4stems",
                                 "spleeter:5stems"]),
              help="Spleeter separation model.")
@click.option("--stems-dir", type=click.Path(exists=True), default=None,
              help="Use pre-separated stems from this folder instead of Spleeter.")
@click.option("--lufs", default=-14.0, show_default=True,
              help="Target master loudness (LUFS).")
@click.option("--brighten", default=6.0, show_default=True,
              help="Max clarity boost in dB for muffled stems.")
@click.option("--no-stems", is_flag=True, help="Only export the final mix.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress output.")
def main(input_path, output_dir, model, stems_dir, lufs, brighten,
         no_stems, quiet):
    """Separate INPUT_PATH into stems, space and clarify them, and remix."""
    cfg = MixConfig(target_lufs=lufs, max_brighten_db=brighten)
    engine = FolderEngine(stems_dir) if stems_dir else SpleeterEngine(model)
    pipeline = MixPipeline(engine=engine, config=cfg)
    out = pipeline.process(input_path, output_dir,
                           export_stems=not no_stems, verbose=not quiet)
    click.echo(out)


if __name__ == "__main__":
    main()
