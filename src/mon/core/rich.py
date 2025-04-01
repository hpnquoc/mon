#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``rich`` for text formatting in terminal, console, and ``mon`` logging."""

from __future__ import annotations

__all__ = [
    "MemoryUsageColumn",
    "ProcessedItemsColumn",
    "ProcessingSpeedColumn",
    "console",
    "error_console",
    "field_style",
    "get_console",
    "get_download_bar",
    "get_error_console",
    "get_progress_bar",
    "get_terminal_size",
    "level_styles",
    "print_dict",
    "print_table",
    "rich_console_theme",
    "set_terminal_size",
]

import fcntl
import shutil
import struct
import subprocess
import sys
import termios

import rich
import torch
from plum import dispatch
from rich import panel, pretty, progress, table, text, theme

from mon.core import enums, type_extensions, utils


# region Console

def get_terminal_size() -> tuple[int, int]:
    """Gets the size of the terminal window in columns and rows.

    Returns:
        Tuple of ``(columns, rows)`` as integers.
    """
    size = shutil.get_terminal_size(fallback=(100, 40))
    return size.columns, size.lines


def set_terminal_size(rows: int = 40, cols: int = 100):
    """Sets the terminal window size to specified rows and columns.

    Args:
        rows: Number of rows for terminal. Default is ``40``.
        cols: Number of columns for terminal. Default is ``100``.
    """
    fd   = sys.stdout.fileno()
    size = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, size)
    subprocess.run(["stty", "rows", str(rows), "cols", str(cols)])


field_style = {
    "asctime"  : {"color": "green"},
    "levelname": {"bold" : True},
    "file_name": {"color": "cyan"},
    "funcName" : {"color": "blue"}
}

level_styles = {
    "critical": {"bold" : True, "color": "red"},
    "debug"   : {"color": "green"},
    "error"   : {"color": "red"},
    "info"    : {"color": "magenta"},
    "warning" : {"color": "yellow"}
}

rich_console_theme = theme.Theme(
    {
        "debug"   : "dark_green",
        "info"    : "green",
        "warning" : "yellow",
        "error"   : "bright_red",
        "critical": "bold red",
    }
)

console = rich.console.Console(
    color_system    = "auto",
    log_time_format = "[%m/%d/%Y %H:%M:%S]",
    soft_wrap       = True,
    width           = get_terminal_size()[0],  # 150
    theme           = rich_console_theme,
)

error_console = rich.console.Console(
    color_system    = "auto",
    log_time_format = "[%m/%d/%Y %H:%M:%S]",
    soft_wrap       = False,
    width           = get_terminal_size()[0],  # 150
    stderr          = True,
    style           = "bold red",
    theme           = rich_console_theme,
)


def get_console() -> rich.console.Console:
    """Gets the global ``rich.console.Console`` object, creating it if needed.

    Returns:
        Global ``rich.console.Console`` instance.
    """
    global console
    if console is None:
        console = rich.console.Console(
            color_system    = "auto",
            log_time_format = "[%m/%d/%Y %H:%M:%S]",
            soft_wrap       = False,
            width           = 150,
            theme           = rich_console_theme,
        )
    return console


def get_error_console() -> rich.console.Console:
    """Gets the global error ``rich.console.Console``, creating it if needed.

    Returns:
        Global ``rich.console.Console`` for error logging.
    """
    global error_console
    if error_console is None:
        error_console = rich.console.Console(
            color_system    = "auto",
            log_time_format = "[%m/%d/%Y %H:%M:%S]",
            soft_wrap       = False,
            width           = 150,
            stderr          = True,
            style           = "bold red",
            theme           = rich_console_theme,
        )
    return error_console

# endregion


# region Progress

def get_download_bar(transient: bool = False, disable: bool = False) -> progress.Progress:
    """Creates a ``rich.progress.Progress`` for download tracking.

    Args:
        transient: If ``True``, hides bar after completion. Default is ``False``.
        disable: If ``True``, disables progress bar. Default is ``False``.

    Returns:
        ``rich.progress.Progress`` with download-specific columns.
    """
    return progress.Progress(
        progress.TextColumn(
            console.get_datetime().strftime("[%x %H:%M:%S]"),
            justify="left",
            style="log.time",
        ),
        progress.TextColumn("{task.description}", justify="right"),
        progress.BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        progress.TransferSpeedColumn(),
        "•",
        progress.DownloadColumn(),
        "•",
        progress.TimeRemainingColumn(),
        ">",
        progress.TimeElapsedColumn(),
        console=console,
        transient=transient,
        disable=disable,
    )


def get_progress_bar(transient: bool = False, disable: bool = False) -> progress.Progress:
    """Creates a ``rich.progress.Progress`` for general progress tracking.

    Args:
        transient: If ``True``, hides bar after completion. Default is ``False``.
        disable: If ``True``, disables progress bar. Default is ``False``.

    Returns:
        ``rich.progress.Progress`` with processing-specific columns.
    """
    return progress.Progress(
        progress.TextColumn(
            console.get_datetime().strftime("[%x %H:%M:%S]"),
            justify="left",
            style="log.time"
        ),
        progress.TextColumn("{task.description}", justify="right"),
        progress.BarColumn(bar_width=None, finished_style="green"),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        ProcessedItemsColumn(),
        "•",
        ProcessingSpeedColumn(),
        "•",
        progress.TimeRemainingColumn(),
        ">",
        progress.TimeElapsedColumn(),
        progress.SpinnerColumn(),
        console=console,
        transient=transient,
        disable=disable,
    )


class MemoryUsageColumn(progress.ProgressColumn):
    """Displays CPU/GPU memory usage in a progress bar (e.g., ``33.1/48.0GB``).

    Args:
        devices: GPU device index or list of indices. Default is ``0``.
        unit: Memory unit (e.g., ``'GB'``). Default is ``MemoryUnit.GB``.
        table_column: Column in table to associate with. Default is ``None``.
    """
    
    def __init__(
        self,
        devices     : int | list[int] = 0,
        unit        : enums.MemoryUnit = enums.MemoryUnit.GB,
        table_column: table.Column    = None
    ):
        super().__init__(table_column=table_column)
        self.devices = type_extensions.to_int_list(devices)
        self.unit    = enums.MemoryUnit.from_value(value=unit)
    
    def render(self, task: progress.Task) -> text.Text:
        """Renders current GPU or CPU memory usage as text.

        Args:
            task: ``rich.progress.Task`` object for the progress task.

        Returns:
            ``rich.text.Text`` with memory usage status.
        """
        return self.get_gpu_memory_text(task) \
            if torch.cuda.is_available() \
            else self.get_machine_memory_text(task)
    
    def get_machine_memory_text(self, task: progress.Task) -> text.Text:
        """Renders current RAM usage as text.

        Args:
            task: ``rich.progress.Task`` object for the progress task.

        Returns:
            ``rich.text.Text`` with RAM usage status.
        """
        total, used, _ = utils.get_machine_memory(unit=self.unit)
        memory_status  = f"{used:.1f}/{total:.1f}{self.unit.value} (CPU)"
        memory_text    = text.Text(memory_status, style="bright_yellow")
        return memory_text
    
    def get_gpu_memory_text(self, task: progress.Task) -> text.Text:
        """Renders current GPU memory usage as text.

        Args:
            task: ``rich.progress.Task`` object for the progress task.

        Returns:
            ``rich.text.Text`` with GPU memory usage status.
        """
        num_devices = len(self.devices)
        totals, useds = [], []
        for i in self.devices:
            total, used, _ = utils.get_gpu_device_memory(device=i, unit=self.unit)
            totals.append(total)
            useds.append(used)
        total = min(totals)
        used  = max(useds)
        memory_status = f"{used:.1f}/{total:.1f}{self.unit.value} ({num_devices} GPUs)"
        memory_text   = text.Text(memory_status, style="bright_yellow")
        return memory_text


class ProcessedItemsColumn(progress.ProgressColumn):
    """Shows number of processed items in a progress bar (e.g., ``1728/2025``).

    Args:
        table_column: Column in table to associate with. Default is ``None``.
    """
    
    def __init__(self, table_column: table.Column = None):
        super().__init__(table_column=table_column)
    
    def render(self, task: progress.Task) -> text.Text:
        """Renders the number of processed items as text.

        Args:
            task: ``rich.progress.Task`` object for the progress task.

        Returns:
            ``rich.text.Text`` with processed items count.
        """
        completed = int(task.completed)
        total     = int(task.total)
        count     = f"{completed}/{total}"
        count     = f"{count:>14}"
        return text.Text(count, style="progress.download")


class ProcessingSpeedColumn(progress.ProgressColumn):
    """Shows human-readable processing speed in a progress bar."""
    
    def render(self, task: progress.Task) -> text.Text:
        """Renders the processing speed as text.

        Args:
            task: ``rich.progress.Task`` object for the progress task.

        Returns:
            ``rich.text.Text`` with the processing speed.
        """
        speed = task.speed
        if speed is None:
            return text.Text("?", style="progress.data.speed")
        speed_text = f"{speed:0.2f}"
        speed_text = f"{speed_text:>7}"
        return text.Text(f"{speed_text}it/s", style="progress.data.speed")

# endregion


# region Print

def print_dict(x: dict, title: str = ""):
    """Prints a dictionary with a title using ``rich.pretty.Pretty`` format.

    Args:
        x: Dictionary to print.
        title: Title above the dictionary. Default is ``""``.

    Raises:
        TypeError: If ``x`` is not a dictionary.
    """
    if not isinstance(x, dict):
        raise TypeError(f"[x] must be a dict, got {type(x).__name__}.")
    pr = pretty.Pretty(
        x,
        expand_all    = True,
        indent_guides = True,
        insert_line   = True,
        overflow      = "fold"
    )
    p = panel.Panel(pr, title=f"{title}")
    console.log(p)


@dispatch
def print_table(x: list[dict]):
    """Prints a list of dictionaries as a ``rich.table.Table``.

    Args:
        x: List of dicts with identical keys to print as a table.

    Raises:
        TypeError: If ``x`` is not a list or has non-dict elements.
        ValueError: If dicts in ``x`` lack identical keys or ``x`` is empty.
    """
    if not isinstance(x, list) or not all(isinstance(d, dict) for d in x):
        raise TypeError(f"[x] must be a list of dicts, got {type(x).__name__}.")
    if not x:
        raise ValueError("[x] must not be empty.")
    if not all(set(d.keys()) == set(x[0].keys()) for d in x):
        raise ValueError("All dictionaries in [x] must have identical keys.")
    tab = table.Table(show_header=True, header_style="bold magenta")
    for k in x[0].keys():
        tab.add_column(k, no_wrap=True)
    for d in x:
        row = [f"{v}" for v in d.values()]
        tab.add_row(*row)
    console.log(tab)


@dispatch
def print_table(x: dict):
    """Prints a dictionary as a ``rich.table.Table``.

    Args:
        x: Dictionary to print as a table.

    Raises:
        TypeError: If ``x`` is not a dictionary.
    """
    if not isinstance(x, dict):
        raise TypeError(f"[x] must be a dict, got {type(x).__name__}.")
    tab = table.Table(show_header=True, header_style="bold magenta")
    tab.add_column("Key")
    tab.add_column("Value")
    for k, v in x.items():
        row = [f"{k}", f"{v}"]
        tab.add_row(*row)
    console.log(tab)

# endregion
