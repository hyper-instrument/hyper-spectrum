"""Typer command surface for JSON-first scientific agents."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from hyperspectrum import agent as services

app = typer.Typer(no_args_is_help=True, help="Agent-ready spectroscopy runtime.")
data_app = typer.Typer(no_args_is_help=True, help="Discover spectroscopy data.")
task_app = typer.Typer(no_args_is_help=True, help="Recommend evidence-backed tasks.")
tools_app = typer.Typer(no_args_is_help=True, help="Match registered tools.")
run_app = typer.Typer(no_args_is_help=True, help="Plan and execute admitted runs.")
app.add_typer(data_app, name="data")
app.add_typer(task_app, name="task")
app.add_typer(tools_app, name="tools")
app.add_typer(run_app, name="run")


@app.command()
def doctor(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit one JSON envelope.")
    ] = False,
) -> None:
    """Inspect local readiness without reading credentials."""

    _invoke(json_output, services.doctor)


@data_app.command("discover")
def data_discover(
    modality: Annotated[str, typer.Option("--modality")],
    profile: Annotated[str, typer.Option("--profile")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit one JSON envelope.")
    ] = False,
) -> None:
    """Discover catalog evidence for a spectroscopy modality."""

    _invoke(json_output, services.discover_data, modality, profile)


@task_app.command("recommend")
def task_recommend(
    candidate_file: Annotated[Path, typer.Option("--candidate-file")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit one JSON envelope.")
    ] = False,
) -> None:
    """Recommend tasks from an explicit candidate artifact."""

    _invoke(json_output, services.recommend_task, candidate_file)


@tools_app.command("match")
def tools_match(
    task: Annotated[str, typer.Option("--task")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit one JSON envelope.")
    ] = False,
) -> None:
    """Match compatible and locally available registered tools."""

    _invoke(json_output, services.match_tools, task)


@run_app.command("plan")
def run_plan(
    task_file: Annotated[Path, typer.Option("--task-file")],
    candidate_file: Annotated[Path, typer.Option("--candidate-file")],
    verdict_file: Annotated[Path, typer.Option("--verdict-file")],
    tool_id: Annotated[str, typer.Option("--tool-id")],
    output_directory: Annotated[Path, typer.Option("--output-directory")],
    max_samples: Annotated[int, typer.Option("--max-samples", min=1)] = 8,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit one JSON envelope.")
    ] = False,
) -> None:
    """Build an immutable, fail-closed local run plan."""

    _invoke(
        json_output,
        services.plan_run,
        task_file=task_file,
        candidate_file=candidate_file,
        verdict_file=verdict_file,
        tool_id=tool_id,
        output_directory=output_directory,
        max_samples=max_samples,
        dry_run=dry_run,
    )


@run_app.command("local")
def run_local(
    plan_file: Annotated[Path, typer.Option("--plan-file")],
    source_npz: Annotated[Path, typer.Option("--source-npz")],
    sample_ids: Annotated[list[str], typer.Option("--sample-id")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit one JSON envelope.")
    ] = False,
) -> None:
    """Execute one admitted local plan without scoring."""

    _invoke(
        json_output,
        services.run_local,
        plan_file=plan_file,
        source_npz=source_npz,
        sample_ids=tuple(sample_ids),
    )


def _invoke(
    json_output: bool,
    service: Callable[..., services.ServiceResponse],
    *args: object,
    **kwargs: object,
) -> None:
    try:
        response = service(*args, **kwargs)
    except services.AgentServiceError as error:
        _emit_error(json_output, error)
        raise typer.Exit(error.exit_code) from error
    except Exception as error:  # pragma: no cover - defensive process boundary
        wrapped = services.AgentExecutionError(str(error))
        _emit_error(json_output, wrapped)
        raise typer.Exit(wrapped.exit_code) from error
    for warning in response.warnings:
        typer.echo(warning, err=True)
    if json_output:
        _emit_envelope(ok=True, result=response.result, warnings=response.warnings)
    else:
        typer.echo(json.dumps(response.result, indent=2, sort_keys=True))


def _emit_error(json_output: bool, error: services.AgentServiceError) -> None:
    typer.echo(str(error), err=True)
    if json_output:
        _emit_envelope(
            ok=False,
            result=error.result,
            error={"code": error.error_code, "message": str(error)},
        )


def _emit_envelope(
    *,
    ok: bool,
    result: object,
    warnings: tuple[str, ...] = (),
    error: dict[str, str] | None = None,
) -> None:
    envelope: dict[str, Any] = {
        "schema_version": "hyperspectrum-cli/v1",
        "ok": ok,
        "result": result,
        "warnings": list(warnings),
        "error": error,
    }
    typer.echo(json.dumps(envelope, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    app()
