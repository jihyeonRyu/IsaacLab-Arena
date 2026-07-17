# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import glob
import os
import sys
import tempfile
import torch
import traceback
from contextlib import nullcontext, suppress

import isaaclab.app as isaaclab_app
import omni.kit.app
from isaaclab.app import AppLauncher


def get_isaac_sim_version() -> str:
    """Get the version of Isaac Sim."""
    return omni.kit.app.get_app().get_app_version()


STARTUP_COMPLETE_MARKER = "[isaaclab-arena] AppLauncher initialization complete"

# The GUI experience (``isaaclab.python.kit``, selected for ``--viz kit`` without cameras) declares a
# dependency on this Isaac Sim extension. During the Isaac Lab Beta 2 upgrade the bundled Isaac Sim
# image does not provide it, so the Kit app fails to start with a dependency-resolution error.
_MISSING_GUI_EXTENSION = "isaacsim.sensors.experimental.rtx"


def _isaaclab_apps_dir() -> str:
    """Return the Isaac Lab ``apps`` directory (matching AppLauncher's own resolution)."""
    return os.path.join(os.path.dirname(os.path.abspath(isaaclab_app.__file__)), *[".."] * 4, "apps")


def _extension_is_available(ext_name: str) -> bool:
    """Return whether a Kit extension directory is present in the Isaac Sim install."""
    exp_path = os.environ.get("EXP_PATH", "")
    if not exp_path:
        return True  # can't tell -> assume present and don't patch
    base = os.path.dirname(exp_path)
    exts_dirs = ("exts", "extscache", os.path.join("kit", "exts"))
    return any(glob.glob(os.path.join(base, exts_dir, ext_name + "*")) for exts_dir in exts_dirs)


def _maybe_patch_gui_experience(args: argparse.Namespace) -> None:
    """Work around the Beta 2 upgrade skew that breaks the ``--viz kit`` GUI experience.

    When the plain GUI experience (``isaaclab.python.kit``) would be selected and it depends on an
    Isaac Sim extension the current image does not provide, write a patched copy with that dependency
    removed and point AppLauncher at it via ``args.experience``. No-op when a custom experience is
    already requested, when the GUI experience is not the one being launched, or on images where the
    extension is present (i.e. after the Isaac Sim image is bumped to match Beta 2).
    """
    if getattr(args, "experience", "") not in ("", None):
        return
    visualizers = getattr(args, "visualizer", None) or []
    is_plain_gui = (
        "kit" in visualizers and not getattr(args, "enable_cameras", False) and not getattr(args, "xr", False)
    )
    if not is_plain_gui:
        return

    source_kit = os.path.join(_isaaclab_apps_dir(), "isaaclab.python.kit")
    if not os.path.isfile(source_kit):
        return
    with open(source_kit, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    if not any(_MISSING_GUI_EXTENSION in ln for ln in lines) or _extension_is_available(_MISSING_GUI_EXTENSION):
        return  # unaffected image; leave the experience untouched

    apps_dir = os.path.dirname(source_kit)
    lab_source = os.path.abspath(os.path.join(apps_dir, "..", "source"))
    patched: list[str] = []
    for ln in lines:
        if _MISSING_GUI_EXTENSION in ln:
            continue  # drop the unresolved dependency
        patched.append(ln)
        if ln.strip().startswith("folders = ["):
            # ${app}-relative ext folders no longer resolve once the .kit lives outside apps/,
            # so add absolute paths to the Isaac Lab apps/ and source/ extension roots.
            patched.append(f'    "{apps_dir}",')
            patched.append(f'    "{lab_source}",')

    fd, patched_path = tempfile.mkstemp(prefix="arena_viz_", suffix=".kit")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(patched) + "\n")
    args.experience = patched_path
    sys.__stderr__.write(
        f"[isaaclab-arena] '{_MISSING_GUI_EXTENSION}' is unavailable in this Isaac Sim image; "
        f"launching --viz kit with a patched experience file: {patched_path}\n"
    )
    sys.__stderr__.flush()


def get_app_launcher(args: argparse.Namespace) -> AppLauncher:
    """Get an app launcher."""
    import time

    _maybe_patch_gui_experience(args)
    t0 = time.monotonic()
    app_launcher = AppLauncher(args)
    elapsed = time.monotonic() - t0
    sys.__stderr__.write(f"{STARTUP_COMPLETE_MARKER} ({elapsed:.1f}s)\n")
    sys.__stderr__.flush()
    return app_launcher


def teardown_simulation_app(suppress_exceptions: bool = False, make_new_stage: bool = True) -> None:
    """
    Tear down the SimulationApp and start a fresh USD stage preparing for the next content.
    Useful for loading new content into the SimulationApp without restarting the app.

    Args:
        suppress_exceptions: Whether to suppress exceptions. If True, the exception will be caught and the execution will continue. If False, the exception will be propagated.
        make_new_stage: Whether to make a new USD stage. If True, a new USD stage will be created. If False, the current USD stage will be used.
    """
    if suppress_exceptions:
        # silently caught exceptions and continue the execution.
        error_manager = suppress(Exception)
    else:
        # Do nothing and let the exception to be raised.
        error_manager = nullcontext()

    with error_manager:
        # Local import to avoid loading Isaac/Kit unless needed.
        from isaaclab.sim import SimulationContext

        sim = None
        with error_manager:
            sim = SimulationContext.instance()

        # Stop the simulation app
        if sim is not None:
            with error_manager:
                # Some versions gate shutdown on this flag.
                sim._disable_app_control_on_stop_handle = True  # noqa: SLF001 (intentional private attr)
            with error_manager:
                sim.stop()
            with error_manager:
                sim.clear_instance()

    # Stop the timeline
    with error_manager:
        import omni.timeline

        with error_manager:
            omni.timeline.get_timeline_interface().stop()

    # Finally, start a fresh USD stage for the next test
    if make_new_stage:
        with error_manager:
            import omni.usd

            omni.usd.get_context().new_stage()


def collect_garbage_and_clear_cuda_cache() -> None:
    """Run GC and release cached CUDA allocations after a sim env is torn down."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reapply_viewer_cfg(env) -> None:
    """Re-apply ViewerCfg camera position after visualizers are initialized.

    ViewportCameraController calls sim.set_camera_view() during __init__, but visualizers
    (e.g. KitVisualizer) are not yet initialized at that point and silently ignore the call.
    After gym.make() returns the visualizers are ready, so we call update_view_location()
    again to apply the configured eye/lookat position.
    """
    unwrapped = env.unwrapped
    vcc = getattr(unwrapped, "viewport_camera_controller", None)
    if vcc is not None:
        vcc.update_view_location()


def _kill_child_processes() -> None:
    """SIGKILL all direct child processes of the current process via /proc."""
    import signal

    my_pid = os.getpid()
    with suppress(FileNotFoundError, PermissionError):
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/status") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            if int(line.split()[1]) == my_pid:
                                os.kill(int(entry.name), signal.SIGKILL)
                            break
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                continue


class SimulationAppContext:
    """Context manager for launching and closing a simulation app."""

    def __init__(self, args: argparse.Namespace):
        """
        Args:
            args (argparse.Namespace): The arguments to the simulation app.
        """
        self.args = args
        self.app_launcher = None

    def is_running(self) -> bool:
        return self.app_launcher.app.is_running()

    def is_exiting(self) -> bool:
        return self.app_launcher.app.is_exiting()

    def __enter__(self):
        self.app_launcher = get_app_launcher(self.args)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing simulation app")
        if exc_type is not None:
            print(f"Exception caught in SimulationAppContext: {exc_type.__name__}: {exc_val}")
            print("Traceback:")
            traceback.print_exception(exc_type, exc_val, exc_tb)
            print("Killing the process without cleaning up")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)

        # When launched as a test subprocess, skip app.close() which can hang
        # indefinitely in Kit's shutdown path.
        if os.environ.get("ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE") == "1":
            print("Force-exiting subprocess (ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1)")
            sys.stdout.flush()
            sys.stderr.flush()
            # SIGKILL orphaned Kit children (shader compiler, GPU workers, …)
            # so they don't hold GPU resources and block the next test subprocess.
            # We target each child individually via /proc to avoid signalling
            # ourselves (Kit installs a C-level SIGTERM handler that overrides
            # Python's SIG_IGN, so os.killpg is not safe here).
            _kill_child_processes()
            os._exit(0)

        # Normal interactive / non-test path: attempt a clean Kit shutdown.
        # app.close() may terminate the process with exit code 0 regardless of
        # errors — see the error branch above for the workaround.
        self.app_launcher.app.close()
