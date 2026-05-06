"""CLI commands for legalbot."""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    if sys.stdout.encoding != "utf-8":
        os.environ["PYTHONIOENCODING"] = "utf-8"
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import typer
from loguru import logger
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from legalbot import __logo__, __version__


class SafeFileHistory(FileHistory):
    """FileHistory subclass that sanitizes surrogate characters on write."""

    def store_string(self, string: str) -> None:
        safe = string.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
        super().store_string(safe)

from legalbot.cli.stream import StreamRenderer, ThinkingSpinner
from legalbot.config.paths import get_workspace_path, is_default_workspace
from legalbot.config.schema import Config
from legalbot.utils.helpers import sync_workspace_templates
from legalbot.utils.restart import (
    consume_restart_notice_from_env,
    format_restart_completed_message,
    should_show_cli_restart_notice,
)

app = typer.Typer(
    name="legalbot",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=f"{__logo__} legalbot - Personal AI Assistant",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI input: prompt_toolkit for editing, paste, history, and display
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # original termios settings, restored on exit


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    import select
    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    from legalbot.config.paths import get_cli_history_path
    history_file = get_cli_history_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=SafeFileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,
    )


def _make_console() -> Console:
    return Console(file=sys.stdout)


def _render_interactive_ansi(render_fn) -> str:
    """Render Rich output to ANSI so prompt_toolkit can print it safely."""
    ansi_console = Console(
        force_terminal=True,
        color_system=console.color_system or "standard",
        width=console.width,
    )
    with ansi_console.capture() as capture:
        render_fn(ansi_console)
    return capture.get()


def _print_agent_response(
    response: str,
    render_markdown: bool,
    metadata: dict | None = None,
) -> None:
    """Render assistant response with consistent terminal styling."""
    c = _make_console()
    content = response or ""
    body = _response_renderable(content, render_markdown, metadata)
    c.print()
    c.print(f"[cyan]{__logo__} legalbot[/cyan]")
    c.print(body)
    c.print()


def _response_renderable(content: str, render_markdown: bool, metadata: dict | None = None):
    """Render plain-text command output without markdown collapsing newlines."""
    if not render_markdown:
        return Text(content)
    if (metadata or {}).get("render_as") == "text":
        return Text(content)
    return Markdown(content)


async def _print_interactive_line(text: str) -> None:
    """Print async interactive updates with prompt_toolkit-safe Rich styling."""
    def _write() -> None:
        ansi = _render_interactive_ansi(
            lambda c: c.print(f"  [dim]↳ {text}[/dim]")
        )
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


async def _print_interactive_response(
    response: str,
    render_markdown: bool,
    metadata: dict | None = None,
) -> None:
    """Print async interactive replies with prompt_toolkit-safe Rich styling."""
    def _write() -> None:
        content = response or ""
        ansi = _render_interactive_ansi(
            lambda c: (
                c.print(),
                c.print(f"[cyan]{__logo__} legalbot[/cyan]"),
                c.print(_response_renderable(content, render_markdown, metadata)),
                c.print(),
            )
        )
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


def _print_cli_progress_line(text: str, thinking: ThinkingSpinner | None) -> None:
    """Print a CLI progress line, pausing the spinner if needed."""
    from contextlib import nullcontext
    with thinking.pause() if thinking else nullcontext():
        console.print(f"  [dim]↳ {text}[/dim]")


async def _print_interactive_progress_line(text: str, thinking: ThinkingSpinner | None) -> None:
    """Print an interactive progress line, pausing the spinner if needed."""
    from contextlib import nullcontext
    with thinking.pause() if thinking else nullcontext():
        await _print_interactive_line(text)


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit."""
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc


def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} legalbot v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """legalbot - Personal AI Assistant."""
    pass


def _merge_missing_defaults(existing: Any, defaults: Any) -> Any:
    """Recursively fill in missing values from defaults without overwriting user config."""
    if not isinstance(existing, dict) or not isinstance(defaults, dict):
        return existing

    merged = dict(existing)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value
        else:
            merged[key] = _merge_missing_defaults(merged[key], value)
    return merged


def _onboard_plugins(config_path: Path) -> None:
    """Inject default config for all discovered channels (built-in + plugins)."""
    import json
    from legalbot.channels.registry import discover_all

    all_channels = discover_all()
    if not all_channels:
        return

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    channels = data.setdefault("channels", {})
    for name, cls in all_channels.items():
        if name not in channels:
            channels[name] = cls.default_config()
        else:
            channels[name] = _merge_missing_defaults(channels[name], cls.default_config())

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _make_provider(config: Config):
    """Create the appropriate LLM provider from config."""
    from legalbot.providers.base import GenerationSettings
    from legalbot.providers.registry import find_by_name

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    spec = find_by_name(provider_name) if provider_name else None
    backend = spec.backend if spec else "openai_compat"

    # --- validation ---
    if backend == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            console.print("[red]Error: Azure OpenAI requires api_key and api_base.[/red]")
            raise typer.Exit(1)
    elif backend == "openai_compat":
        needs_key = not (p and p.api_key)
        exempt = spec and (spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            console.print("[red]Error: No API key configured.[/red]")
            console.print("Set one in ~/.legalbot/config.json under providers section")
            raise typer.Exit(1)

    # --- instantiation by backend ---
    if backend == "azure_openai":
        from legalbot.providers.azure_openai_provider import AzureOpenAIProvider
        provider = AzureOpenAIProvider(
            api_key=p.api_key,
            api_base=p.api_base,
            default_model=model,
        )
    elif backend == "anthropic":
        from legalbot.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    else:
        from legalbot.providers.openai_compat_provider import OpenAICompatProvider
        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider


def _load_runtime_config(config: str | None = None, workspace: str | None = None) -> Config:
    """Load config and optionally override the active workspace."""
    from legalbot.config.loader import load_config, resolve_config_env_vars, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            raise typer.Exit(1)
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")

    try:
        loaded = resolve_config_env_vars(load_config(config_path))
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    _warn_deprecated_config_keys(config_path)
    if workspace:
        loaded.agents.defaults.workspace = workspace
    return loaded


def _warn_deprecated_config_keys(config_path: Path | None) -> None:
    """Hint users to remove obsolete keys from their config file."""
    import json
    from legalbot.config.loader import get_config_path

    path = config_path or get_config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if "memoryWindow" in raw.get("agents", {}).get("defaults", {}):
        console.print(
            "[dim]Hint: `memoryWindow` in your config is no longer used "
            "and can be safely removed.[/dim]"
        )


def _migrate_cron_store(config: "Config") -> None:
    """One-time migration: move legacy global cron store into the workspace."""
    from legalbot.config.paths import get_cron_dir

    legacy_path = get_cron_dir() / "jobs.json"
    new_path = config.workspace_path / "cron" / "jobs.json"
    if legacy_path.is_file() and not new_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.move(str(legacy_path), str(new_path))


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard(
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Initialize legalbot configuration and workspace."""
    from legalbot.config.loader import get_config_path, load_config, save_config, set_config_path

    if config:
        config_path = Path(config).expanduser().resolve()
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")
    else:
        config_path = get_config_path()

    def _apply_workspace_override(loaded: Config) -> Config:
        if workspace:
            loaded.agents.defaults.workspace = workspace
        return loaded

    # Create or update config
    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print(
            "  [bold]y[/bold] = overwrite with defaults (existing values will be lost)"
        )
        console.print(
            "  [bold]N[/bold] = refresh config, keeping existing values and adding new fields"
        )
        if typer.confirm("Overwrite?"):
            cfg = _apply_workspace_override(Config())
            save_config(cfg, config_path)
            console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
        else:
            cfg = _apply_workspace_override(load_config(config_path))
            save_config(cfg, config_path)
            console.print(
                f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved)"
            )
    else:
        cfg = _apply_workspace_override(Config())
        save_config(cfg, config_path)
        console.print(f"[green]✓[/green] Created config at {config_path}")

    _onboard_plugins(config_path)

    # Create workspace
    workspace_path = get_workspace_path(cfg.workspace_path)
    if not workspace_path.exists():
        workspace_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace_path}")

    sync_workspace_templates(workspace_path)

    agent_cmd = 'legalbot agent -m "Hello!"'
    gateway_cmd = "legalbot gateway"
    console.print(f"\n{__logo__} legalbot is ready!")
    console.print("\nNext steps:")
    console.print(f"  1. Add your API key to [cyan]{config_path}[/cyan]")
    console.print("     Get one at: https://openrouter.ai/keys")
    console.print(f"  2. Chat: [cyan]{agent_cmd}[/cyan]")
    console.print(f"  3. Gateway: [cyan]{gateway_cmd}[/cyan]")


# ============================================================================
# OpenAI-Compatible API Server
# ============================================================================


@app.command()
def serve(
    port: int | None = typer.Option(None, "--port", "-p", help="API server port"),
    host: str | None = typer.Option(None, "--host", "-H", help="Bind address"),
    timeout: float | None = typer.Option(None, "--timeout", "-t", help="Per-request timeout (seconds)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show legalbot runtime logs"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Start the OpenAI-compatible API server (/v1/chat/completions)."""
    try:
        from aiohttp import web  # noqa: F401
    except ImportError:
        console.print("[red]aiohttp is required. Install with: pip install 'legalbot-ai[api]'[/red]")
        raise typer.Exit(1)

    from legalbot.agent.loop import AgentLoop
    from legalbot.api.server import create_app
    from legalbot.bus.queue import MessageBus
    from legalbot.session.manager import SessionManager

    if verbose:
        logger.enable("legalbot")
    else:
        logger.disable("legalbot")

    runtime_config = _load_runtime_config(config, workspace)
    api_cfg = runtime_config.api
    host = host if host is not None else api_cfg.host
    port = port if port is not None else api_cfg.port
    timeout = timeout if timeout is not None else api_cfg.timeout
    sync_workspace_templates(runtime_config.workspace_path)
    bus = MessageBus()
    provider = _make_provider(runtime_config)
    session_manager = SessionManager(runtime_config.workspace_path)
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=runtime_config.workspace_path,
        model=runtime_config.agents.defaults.model,
        max_iterations=runtime_config.agents.defaults.max_tool_iterations,
        context_window_tokens=runtime_config.agents.defaults.context_window_tokens,
        context_block_limit=runtime_config.agents.defaults.context_block_limit,
        max_tool_result_chars=runtime_config.agents.defaults.max_tool_result_chars,
        provider_retry_mode=runtime_config.agents.defaults.provider_retry_mode,
        web_config=runtime_config.tools.web,
        exec_config=runtime_config.tools.exec,
        restrict_to_workspace=runtime_config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=runtime_config.tools.mcp_servers,
        channels_config=runtime_config.channels,
        timezone=runtime_config.agents.defaults.timezone,
        unified_session=runtime_config.agents.defaults.unified_session,
        rag_config=runtime_config.tools.rag,
        orchestrate_config=runtime_config.tools.orchestrate,
        feedback_config=runtime_config.tools.feedback,
        document_draft_config=runtime_config.tools.document_draft,
        audit_config=runtime_config.tools.audit,
    )

    model_name = runtime_config.agents.defaults.model
    console.print(f"{__logo__} Starting OpenAI-compatible API server")
    console.print(f"  [cyan]Endpoint[/cyan] : http://{host}:{port}/v1/chat/completions")
    console.print(f"  [cyan]Model[/cyan]    : {model_name}")
    console.print("  [cyan]Session[/cyan]  : api:default")
    console.print(f"  [cyan]Timeout[/cyan]  : {timeout}s")
    if host in {"0.0.0.0", "::"}:
        console.print(
            "[yellow]Warning:[/yellow] API is bound to all interfaces. "
            "Only do this behind a trusted network boundary, firewall, or reverse proxy."
        )
    console.print()

    api_app = create_app(agent_loop, model_name=model_name, request_timeout=timeout)

    async def on_startup(_app):
        await agent_loop._connect_mcp()

    async def on_cleanup(_app):
        await agent_loop.close_mcp()

    api_app.on_startup.append(on_startup)
    api_app.on_cleanup.append(on_cleanup)

    web.run_app(api_app, host=host, port=port, print=lambda msg: logger.info(msg))


# ============================================================================
# Gateway / Server
# ============================================================================


@app.command()
def gateway(
    port: int | None = typer.Option(None, "--port", "-p", help="Gateway port"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Start the legalbot gateway."""
    from legalbot.agent.loop import AgentLoop
    from legalbot.bus.queue import MessageBus
    from legalbot.channels.manager import ChannelManager
    from legalbot.cron.service import CronService
    from legalbot.cron.types import CronJob
    from legalbot.session.manager import SessionManager

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    config = _load_runtime_config(config, workspace)
    port = port if port is not None else config.gateway.port

    console.print(f"{__logo__} Starting legalbot gateway version {__version__} on port {port}...")
    sync_workspace_templates(config.workspace_path)
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)

    if is_default_workspace(config.workspace_path):
        _migrate_cron_store(config)

    # Create cron service with workspace-scoped store
    cron_store_path = config.workspace_path / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # Create agent with cron service
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        web_config=config.tools.web,
        context_block_limit=config.agents.defaults.context_block_limit,
        max_tool_result_chars=config.agents.defaults.max_tool_result_chars,
        provider_retry_mode=config.agents.defaults.provider_retry_mode,
        exec_config=config.tools.exec,
        cron_service=cron,
        rag_config=config.tools.rag,
        orchestrate_config=config.tools.orchestrate,
        feedback_config=config.tools.feedback,
        document_draft_config=config.tools.document_draft,
        audit_config=config.tools.audit,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        timezone=config.agents.defaults.timezone,
        unified_session=config.agents.defaults.unified_session,
    )

    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent."""
        from legalbot.agent.tools.cron import CronTool
        from legalbot.agent.tools.message import MessageTool

        reminder_note = (
            "[Scheduled Task] Timer finished.\n\n"
            f"Task '{job.name}' has been triggered.\n"
            f"Scheduled instruction: {job.payload.message}"
        )

        cron_tool = agent.tools.get("cron")
        cron_token = None
        if isinstance(cron_tool, CronTool):
            cron_token = cron_tool.set_cron_context(True)
        try:
            resp = await agent.process_direct(
                reminder_note,
                session_key=f"cron:{job.id}",
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to or "direct",
            )
        finally:
            if isinstance(cron_tool, CronTool) and cron_token is not None:
                cron_tool.reset_cron_context(cron_token)

        response = resp.content if resp else ""

        message_tool = agent.tools.get("message")
        if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
            return response

        if job.payload.deliver and job.payload.to and response:
            from legalbot.bus.events import OutboundMessage
            await bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to,
                content=response,
            ))
        return response

    cron.on_job = on_cron_job

    # Create channel manager
    channels = ChannelManager(config, bus)

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    async def run():
        try:
            await cron.start()
            await asyncio.gather(
                agent.run(),
                channels.start_all(),
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        except Exception:
            import traceback
            console.print("\n[red]Error: Gateway crashed unexpectedly[/red]")
            console.print(traceback.format_exc())
        finally:
            await agent.close_mcp()
            cron.stop()
            agent.stop()
            await channels.stop_all()

    asyncio.run(run())


# ============================================================================
# Agent Commands
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show legalbot runtime logs during chat"),
):
    """Interact with the agent directly."""
    from legalbot.agent.loop import AgentLoop
    from legalbot.bus.queue import MessageBus
    from legalbot.cron.service import CronService

    config = _load_runtime_config(config, workspace)
    sync_workspace_templates(config.workspace_path)

    bus = MessageBus()
    provider = _make_provider(config)

    if is_default_workspace(config.workspace_path):
        _migrate_cron_store(config)

    cron_store_path = config.workspace_path / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    if logs:
        logger.enable("legalbot")
    else:
        logger.disable("legalbot")

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        web_config=config.tools.web,
        context_block_limit=config.agents.defaults.context_block_limit,
        max_tool_result_chars=config.agents.defaults.max_tool_result_chars,
        provider_retry_mode=config.agents.defaults.provider_retry_mode,
        exec_config=config.tools.exec,
        cron_service=cron,
        rag_config=config.tools.rag,
        orchestrate_config=config.tools.orchestrate,
        feedback_config=config.tools.feedback,
        document_draft_config=config.tools.document_draft,
        audit_config=config.tools.audit,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        timezone=config.agents.defaults.timezone,
        unified_session=config.agents.defaults.unified_session,
    )
    restart_notice = consume_restart_notice_from_env()
    if restart_notice and should_show_cli_restart_notice(restart_notice, session_id):
        _print_agent_response(
            format_restart_completed_message(restart_notice.started_at_raw),
            render_markdown=False,
        )

    # Shared reference for progress callbacks
    _thinking: ThinkingSpinner | None = None

    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        ch = agent_loop.channels_config
        if ch and tool_hint and not ch.send_tool_hints:
            return
        if ch and not tool_hint and not ch.send_progress:
            return
        _print_cli_progress_line(content, _thinking)

    if message:
        # Single message mode
        async def run_once():
            renderer = StreamRenderer(render_markdown=markdown)
            response = await agent_loop.process_direct(
                message, session_id,
                on_progress=_cli_progress,
                on_stream=renderer.on_delta,
                on_stream_end=renderer.on_end,
            )
            if not renderer.streamed:
                await renderer.close()
                _print_agent_response(
                    response.content if response else "",
                    render_markdown=markdown,
                    metadata=response.metadata if response else None,
                )
            await agent_loop.close_mcp()

        asyncio.run(run_once())
    else:
        # Interactive mode
        from legalbot.bus.events import InboundMessage
        _init_prompt_session()
        console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")

        if ":" in session_id:
            cli_channel, cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, cli_chat_id = "cli", session_id

        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            _restore_terminal()
            console.print(f"\nReceived {sig_name}, goodbye!")
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, _handle_signal)
        if hasattr(signal, 'SIGPIPE'):
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        async def run_interactive():
            bus_task = asyncio.create_task(agent_loop.run())
            turn_done = asyncio.Event()
            turn_done.set()
            turn_response: list[tuple[str, dict]] = []
            renderer: StreamRenderer | None = None

            async def _consume_outbound():
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

                        if msg.metadata.get("_stream_delta"):
                            if renderer:
                                await renderer.on_delta(msg.content)
                            continue
                        if msg.metadata.get("_stream_end"):
                            if renderer:
                                await renderer.on_end(
                                    resuming=msg.metadata.get("_resuming", False),
                                )
                            continue
                        if msg.metadata.get("_streamed"):
                            turn_done.set()
                            continue

                        if msg.metadata.get("_progress"):
                            is_tool_hint = msg.metadata.get("_tool_hint", False)
                            ch = agent_loop.channels_config
                            if ch and is_tool_hint and not ch.send_tool_hints:
                                pass
                            elif ch and not is_tool_hint and not ch.send_progress:
                                pass
                            else:
                                await _print_interactive_progress_line(msg.content, _thinking)
                            continue

                        if not turn_done.is_set():
                            if msg.content:
                                turn_response.append((msg.content, dict(msg.metadata or {})))
                            turn_done.set()
                        elif msg.content:
                            await _print_interactive_response(
                                msg.content,
                                render_markdown=markdown,
                                metadata=msg.metadata,
                            )

                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            outbound_task = asyncio.create_task(_consume_outbound())

            try:
                while True:
                    try:
                        _flush_pending_tty_input()
                        if renderer:
                            renderer.stop_for_input()
                        user_input = await _read_interactive_input_async()
                        command = user_input.strip()
                        if not command:
                            continue

                        if _is_exit_command(command):
                            _restore_terminal()
                            console.print("\nGoodbye!")
                            break

                        turn_done.clear()
                        turn_response.clear()
                        renderer = StreamRenderer(render_markdown=markdown)

                        await bus.publish_inbound(InboundMessage(
                            channel=cli_channel,
                            sender_id="user",
                            chat_id=cli_chat_id,
                            content=user_input,
                            metadata={"_wants_stream": True},
                        ))

                        await turn_done.wait()

                        if turn_response:
                            content, meta = turn_response[0]
                            if content and not meta.get("_streamed"):
                                if renderer:
                                    await renderer.close()
                                _print_agent_response(
                                    content, render_markdown=markdown, metadata=meta,
                                )
                        elif renderer and not renderer.streamed:
                            await renderer.close()
                    except KeyboardInterrupt:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
            finally:
                agent_loop.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, return_exceptions=True)
                await agent_loop.close_mcp()

        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status(
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Show channel status."""
    from legalbot.channels.registry import discover_all
    from legalbot.config.loader import load_config, set_config_path

    resolved_config_path = Path(config_path).expanduser().resolve() if config_path else None
    if resolved_config_path is not None:
        set_config_path(resolved_config_path)

    config = load_config(resolved_config_path)

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled")

    for name, cls in sorted(discover_all().items()):
        section = getattr(config.channels, name, None)
        if section is None:
            enabled = False
        elif isinstance(section, dict):
            enabled = section.get("enabled", False)
        else:
            enabled = getattr(section, "enabled", False)
        table.add_row(
            cls.display_name,
            "[green]\u2713[/green]" if enabled else "[dim]\u2717[/dim]",
        )

    console.print(table)


@channels_app.command("login")
def channels_login(
    channel_name: str = typer.Argument(..., help="Channel name (e.g. dingtalk, feishu, qq)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-authentication even if already logged in"),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Authenticate with a channel via QR code or other interactive login."""
    from legalbot.channels.registry import discover_all
    from legalbot.config.loader import load_config, set_config_path

    resolved_config_path = Path(config_path).expanduser().resolve() if config_path else None
    if resolved_config_path is not None:
        set_config_path(resolved_config_path)

    config = load_config(resolved_config_path)
    channel_cfg = getattr(config.channels, channel_name, None) or {}

    all_channels = discover_all()
    if channel_name not in all_channels:
        available = ", ".join(all_channels.keys())
        console.print(f"[red]Unknown channel: {channel_name}[/red]  Available: {available}")
        raise typer.Exit(1)

    console.print(f"{__logo__} {all_channels[channel_name].display_name} Login\n")

    channel_cls = all_channels[channel_name]
    channel = channel_cls(channel_cfg, bus=None)

    success = asyncio.run(channel.login(force=force))

    if not success:
        raise typer.Exit(1)


# ============================================================================
# Plugin Commands
# ============================================================================


plugins_app = typer.Typer(help="Manage channel plugins")
app.add_typer(plugins_app, name="plugins")


@plugins_app.command("list")
def plugins_list():
    """List all discovered channels (built-in and plugins)."""
    from legalbot.channels.registry import discover_all, discover_channel_names
    from legalbot.config.loader import load_config

    config = load_config()
    builtin_names = set(discover_channel_names())
    all_channels = discover_all()

    table = Table(title="Channel Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Enabled")

    for name in sorted(all_channels):
        cls = all_channels[name]
        source = "builtin" if name in builtin_names else "plugin"
        section = getattr(config.channels, name, None)
        if section is None:
            enabled = False
        elif isinstance(section, dict):
            enabled = section.get("enabled", False)
        else:
            enabled = getattr(section, "enabled", False)
        table.add_row(
            cls.display_name,
            source,
            "[green]yes[/green]" if enabled else "[dim]no[/dim]",
        )

    console.print(table)


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def status():
    """Show legalbot status."""
    from legalbot.config.loader import get_config_path, load_config

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} legalbot Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        from legalbot.providers.registry import PROVIDERS

        console.print(f"Model: {config.agents.defaults.model}")

        for spec in PROVIDERS:
            p = getattr(config.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_local:
                if p.api_base:
                    console.print(f"{spec.label}: [green]✓ {p.api_base}[/green]")
                else:
                    console.print(f"{spec.label}: [dim]not set[/dim]")
            else:
                has_key = bool(p.api_key)
                console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


if __name__ == "__main__":
    app()


# ============================================================================
# Legal Knowledge Base Commands
# ============================================================================

legal_app = typer.Typer(help="Manage legal knowledge base")
app.add_typer(legal_app, name="legal")


@legal_app.command("index")
def legal_index(
    data_dir: str = typer.Option("./legal_data", "--data-dir", help="Directory containing legal documents"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild index from scratch"),
) -> None:
    """Build or update the legal knowledge base index."""
    import asyncio
    from pathlib import Path

    from legalbot.config.loader import load_config
    from legalbot.rag import create_retriever
    from legalbot.rag.chunker import LegalChunker
    from legalbot.rag.indexer import LegalIndexer
    from legalbot.rag.loader import LegalDocumentLoader

    config = load_config()
    rag_config = config.tools.rag

    if not rag_config.enable:
        console.print("[yellow]RAG is not enabled. Set tools.rag.enable = true in config.[/yellow]")
        raise typer.Exit(1)

    if not rag_config.embedding_api_key:
        console.print("[yellow]Embedding API key not set. Configure tools.rag.embedding_api_key.[/yellow]")
        raise typer.Exit(1)

    data_path = Path(data_dir)
    if not data_path.exists():
        console.print(f"[red]Data directory not found: {data_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"Building legal index from [cyan]{data_dir}[/cyan]...")
    if rebuild:
        console.print("[yellow]Rebuild mode: clearing existing index[/yellow]")

    async def _build():
        retriever = create_retriever(rag_config)
        loader = LegalDocumentLoader()
        chunker = LegalChunker(
            max_chunk_tokens=rag_config.chunk_max_tokens,
            overlap_tokens=rag_config.chunk_overlap_tokens,
        )
        persist_dir = Path(rag_config.persist_dir).expanduser()
        indexer = LegalIndexer(loader, chunker, retriever, persist_dir)
        return await indexer.build_index(data_path, rebuild=rebuild)

    stats = asyncio.run(_build())

    console.print(f"\n[green]Indexing complete[/green]")
    console.print(f"  Documents: {stats.total_documents} total, {stats.new_documents} new, {stats.skipped_documents} skipped")
    console.print(f"  Chunks: {stats.new_chunks} indexed")
    if stats.errors:
        console.print(f"  [red]Errors: {len(stats.errors)}[/red]")
        for err in stats.errors[:5]:
            console.print(f"    - {err}")


@legal_app.command("index-status")
def legal_index_status() -> None:
    """Show the current legal knowledge base index status."""
    from pathlib import Path

    from legalbot.config.loader import load_config
    from legalbot.rag.indexer import LegalIndexer

    config = load_config()
    rag_config = config.tools.rag

    persist_dir = Path(rag_config.persist_dir).expanduser()
    # Create a minimal indexer just to read manifest
    from legalbot.rag.loader import LegalDocumentLoader
    from legalbot.rag.chunker import LegalChunker

    loader = LegalDocumentLoader()
    chunker = LegalChunker()
    # We don't need a real retriever just for status
    indexer = LegalIndexer(loader, chunker, None, persist_dir)  # type: ignore[arg-type]

    status = indexer.get_status()

    console.print(f"[cyan]Legal Knowledge Base Status[/cyan]")
    console.print(f"  Persist dir: {status['persist_dir']}")
    console.print(f"  Indexed documents: {status['indexed_documents']}")
    console.print(f"  Total chunks: {status['total_chunks']}")

    if status["documents"]:
        console.print(f"\n  Documents:")
        for doc_key, info in status["documents"].items():
            console.print(f"    - {doc_key}: {info.get('chunk_count', 0)} chunks (indexed {info.get('indexed_at', 'unknown')})")


# Feedback Commands
# ============================================================================

feedback_app = typer.Typer(help="Manage legal RAG feedback")
app.add_typer((feedback_app), name="feedback")


@feedback_app.command("list")
def feedback_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of records to show"),
    feedback_type: str | None = typer.Option(None, "--type", help="Filter by type: helpful/unhelpful/correction"),
) -> None:
    """List recent feedback records."""
    from legalbot.feedback import FeedbackStorage

    storage = FeedbackStorage()
    records = storage.list_records(limit=limit, feedback_type=feedback_type)

    if not records:
        console.print("[yellow]No feedback records found.[/yellow]")
        return

    table = Table(title=f"Recent Feedback ({len(records)} records)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Time", style="dim")
    table.add_column("Type")
    table.add_column("Query", max_width=40)
    table.add_column("Result ID", style="dim")

    for r in records:
        table.add_row(
            r.id,
            r.timestamp.strftime("%m-%d %H:%M"),
            r.type,
            r.query.text[:40],
            r.results[0].chunk_id[:12] if r.results else r.correction.chunk_id[:12] if r.correction else "",
        )

    console.print(table)


@feedback_app.command("export")
def feedback_export(
    from_date: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD)"),
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    feedback_type: str | None = typer.Option(None, "--type", help="Filter by type"),
) -> None:
    """Export feedback data to a JSONL file."""
    import json
    from datetime import datetime

    from legalbot.feedback import FeedbackStorage

    since = datetime.strptime(from_date, "%Y-%m-%d")
    until = datetime.strptime(to_date, "%Y-%m-%d")

    storage = FeedbackStorage()
    records = storage.query(since=since, until=until, feedback_type=feedback_type)

    with open(output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    console.print(f"[green]Exported {len(records)} records to {output}[/green]")


@feedback_app.command("analyze")
def feedback_analyze(
    period: str = typer.Option("week", "--period", help="Analysis period: day/week/month"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file for report"),
) -> None:
    """Analyze feedback and generate an improvement report."""
    from datetime import timedelta

    from legalbot.feedback import FeedbackAnalyzer, FeedbackStorage

    today = datetime.now()
    if period == "day":
        since = today - timedelta(days=1)
    elif period == "week":
        since = today - timedelta(weeks=1)
    else:
        since = today - timedelta(days=30)
    until = today

    storage = FeedbackStorage()
    analyzer = FeedbackAnalyzer(storage)

    async def _analyze():
        return await analyzer.analyze_period(since, until)

    import asyncio
    report = asyncio.run(_analyze())
    markdown = analyzer.generate_markdown_report(report)

    if output:
        Path(output).write_text(markdown, encoding="utf-8")
        console.print(f"[green]Report written to {output}[/green]")
    else:
        console.print(Markdown(markdown))


# Audit Commands
# ============================================================================

audit_app = typer.Typer(help="Legal audit log query and management")
app.add_typer(audit_app, name="audit")


@audit_app.command("query")
def audit_query(
    event_type: str | None = typer.Option(None, "--type", "-t", help="Filter by event type"),
    session_id: str | None = typer.Option(None, "--session", "-s", help="Filter by session ID"),
    start_date: str | None = typer.Option(None, "--from", help="Start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--to", help="End date (YYYY-MM-DD)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max records to return"),
) -> None:
    """Query the legal audit log with optional filters."""
    import asyncio
    from legalbot.audit import LegalAuditLogger
    from rich.table import Table

    logger = LegalAuditLogger()

    async def _query():
        return await logger.query(
            start_date=start_date,
            end_date=end_date,
            event_type=event_type,
            session_id=session_id,
            limit=limit,
        )

    records = asyncio.run(_query())

    if not records:
        console.print("[yellow]No audit records found.[/yellow]")
        return

    table = Table(title=f"Audit Log ({len(records)} records)")
    table.add_column("Event ID", style="cyan", no_wrap=True, max_width=12)
    table.add_column("Time", style="dim", max_width=16)
    table.add_column("Type", max_width=18)
    table.add_column("Session", max_width=20)
    table.add_column("Query", max_width=30)

    for r in records:
        query_text = r.get("query", {}).get("original_text", "")[:30]
        table.add_row(
            r.get("event_id", "")[:12],
            r.get("timestamp", "")[:16],
            r.get("event_type", ""),
            r.get("session_id", "")[:20],
            query_text,
        )

    console.print(table)


@audit_app.command("cleanup")
def audit_cleanup(
    retention_days: int = typer.Option(90, "--retention", "-r", help="Retention period in days"),
) -> None:
    """Delete audit log files older than retention period."""
    import asyncio
    from legalbot.audit import LegalAuditLogger

    logger = LegalAuditLogger()

    async def _cleanup():
        return await logger.cleanup_old_logs()

    deleted = asyncio.run(_cleanup())
    console.print(f"[green]Deleted {deleted} old audit log file(s).[/green]")


@audit_app.command("verify")
def audit_verify(
    event_id: str | None = typer.Option(None, "--event-id", help="Verify specific event ID"),
) -> None:
    """Verify integrity of audit log records (hash check)."""
    import asyncio
    from legalbot.audit import LegalAuditLogger

    logger = LegalAuditLogger()

    async def _verify():
        return await logger.verify_integrity(event_id=event_id)

    result = asyncio.run(_verify())

    if result.get("corrupted"):
        console.print(f"[red]Integrity check FAILED: {len(result['corrupted'])} corrupted record(s)[/red]")
        for cid in result["corrupted"]:
            console.print(f"  - {cid}")
    else:
        console.print(
            f"[green]Integrity check passed: {result['valid']}/{result['checked']} record(s) valid[/green]"
        )
    if "note" in result:
        console.print(f"[yellow]{result['note']}[/yellow]")

