"""
imbdata.cli — Command-line interface.

Exposes the package's operations as the ``imbdata`` console script:

* ``imbdata list [--domain D]``      — list registered datasets;
* ``imbdata info KEY``               — show a dataset's metadata;
* ``imbdata download KEY... | --all``— materialize datasets into the store;
* ``imbdata verify [KEY...]``        — check cached files against the manifest;
* ``imbdata status``                 — summarize the store's location and size.

The subcommands are methods of :class:`CLI`, which owns the argument parser and
the :class:`~imbdata.api.DatasetService` they operate on.

Author:
    Luis García Rodríguez
    Doctorado en Ciencia e Ingeniería de la Computación (DCIC)
    IIMAS — Universidad Nacional Autónoma de México (UNAM)
    CVU: 905206 · ORCID: 0009-0004-9514-5508

Project:
    imbdata v0.2.0 — Imbalanced Classification Dataset Repository
    Advisor: Dr. José Antonio Neme Castillo
    Research Group: Anomalocaris
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from imbdata import __version__
from imbdata.api import DatasetService
from imbdata.exceptions import ImbdataError
from imbdata.verify import STATUS_OK

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1

__all__ = ["CLI", "main"]


class CLI:
    """Argument parsing and command dispatch for the ``imbdata`` console script.

    Attributes:
        service: The dataset service every subcommand operates on.
        parser: The configured top-level argument parser.

    Example:
        >>> CLI().run(["list", "--domain", "medicine"])
        0
    """

    def __init__(self, service: DatasetService | None = None) -> None:
        """Initialize the CLI.

        Args:
            service: Dataset service to operate on. Defaults to a new service
                bound to the resolved data store.
        """
        self.service = service if service is not None else DatasetService()
        self.parser = self.build_parser()

    # ── Parser ─────────────────────────────────────────────────────────

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Construct the top-level parser and its subcommands.

        Returns:
            The configured :class:`argparse.ArgumentParser`.
        """
        parser = argparse.ArgumentParser(
            prog="imbdata",
            description="Centralized repository of imbalanced classification datasets.",
        )
        parser.add_argument("--version", action="version", version=f"imbdata {__version__}")
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="show INFO-level progress messages",
        )
        subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

        listing = subparsers.add_parser("list", help="list registered datasets")
        listing.add_argument("--domain", help="restrict the listing to one domain")
        listing.add_argument(
            "--cached",
            action="store_true",
            help="list only datasets already materialized in the store",
        )

        details = subparsers.add_parser("info", help="show a dataset's metadata")
        details.add_argument("name", help="dataset key")
        details.add_argument("--variant", help="optional dataset variant")

        download = subparsers.add_parser(
            "download", help="download and preprocess datasets into the store"
        )
        download.add_argument("names", nargs="*", help="dataset keys")
        download.add_argument("--domain", help="download every dataset of this domain")
        download.add_argument("--all", action="store_true", help="download every dataset")
        download.add_argument(
            "--force", action="store_true", help="rebuild datasets that are already cached"
        )

        verification = subparsers.add_parser(
            "verify", help="check cached datasets against the manifest"
        )
        verification.add_argument("names", nargs="*", help="dataset keys (default: all)")

        subparsers.add_parser("status", help="show the store path, size, and contents")
        return parser

    # ── Entry point ────────────────────────────────────────────────────

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Parse arguments and dispatch to the requested subcommand.

        Args:
            argv: Argument vector, excluding the program name. Defaults to
                ``sys.argv[1:]``.

        Returns:
            A process exit code: ``0`` on success, ``1`` on failure.
        """
        args = self.parser.parse_args(argv)
        logging.basicConfig(
            level=logging.INFO if args.verbose else logging.WARNING,
            format="%(levelname)s: %(message)s",
        )

        if not args.command:
            self.parser.print_help()
            return EXIT_OK

        handler = getattr(self, f"cmd_{args.command}")
        try:
            return handler(args)
        except ImbdataError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        except KeyboardInterrupt:
            print("interrupted", file=sys.stderr)
            return EXIT_ERROR

    # ── Subcommands ────────────────────────────────────────────────────

    def cmd_list(self, args: argparse.Namespace) -> int:
        """Print the registered datasets, one per line, with their domain.

        Args:
            args: Parsed arguments carrying ``domain`` and ``cached``.

        Returns:
            The process exit code.
        """
        names = self.service.list_datasets(domain=args.domain)
        if args.cached:
            names = [
                name for name in names
                if self.service.config.processed_path(name).is_file()
            ]
        if not names:
            print("no datasets match the given filters")
            return EXIT_OK

        width = max(len(name) for name in names)
        for name in names:
            meta = self.service.registry.get(name)
            cached = "cached" if self.service.config.processed_path(name).is_file() else "-"
            print(f"{name:<{width}}  {meta.get('domain', ''):<22} {cached}")
        print(f"\n{len(names)} dataset(s)")
        return EXIT_OK

    def cmd_info(self, args: argparse.Namespace) -> int:
        """Print a dataset's metadata as aligned key/value lines.

        Args:
            args: Parsed arguments carrying ``name`` and ``variant``.

        Returns:
            The process exit code.
        """
        details = self.service.info(args.name, args.variant)
        width = max(len(key) for key in details)
        for key, value in details.items():
            print(f"{key:<{width}} : {value}")
        return EXIT_OK

    def cmd_download(self, args: argparse.Namespace) -> int:
        """Materialize the requested datasets and report each outcome.

        Args:
            args: Parsed arguments carrying ``names``, ``domain``, ``all``,
                and ``force``.

        Returns:
            ``0`` when every requested dataset succeeded, ``1`` otherwise.
        """
        if args.all:
            names = self.service.list_datasets()
        elif args.domain:
            names = self.service.list_datasets(domain=args.domain)
        else:
            names = list(args.names)

        if not names:
            print("error: give dataset keys, --domain, or --all", file=sys.stderr)
            return EXIT_ERROR

        results = self.service.ensure(names, force=args.force)
        width = max(len(name) for name in results)
        failures = 0
        for name, outcome in results.items():
            marker = "x" if outcome.startswith("ERROR") else "OK"
            failures += outcome.startswith("ERROR")
            print(f"[{marker}] {name:<{width}}  {outcome}")
        print(f"\n{len(results) - failures}/{len(results)} dataset(s) ready")
        return EXIT_ERROR if failures else EXIT_OK

    def cmd_verify(self, args: argparse.Namespace) -> int:
        """Verify cached datasets against the manifest and report each status.

        Args:
            args: Parsed arguments carrying an optional list of ``names``.

        Returns:
            ``0`` when nothing is corrupt, ``1`` when a hash mismatch is found.
        """
        report = self.service.verify(list(args.names) or None)
        width = max(len(name) for name in report) if report else 1
        mismatches = 0
        for name, status in report.items():
            if status == "MISSING":
                continue
            mismatches += status == "HASH_MISMATCH"
            print(f"{name:<{width}}  {status}")

        checked = [s for s in report.values() if s != "MISSING"]
        print(f"\n{sum(s == STATUS_OK for s in checked)}/{len(checked)} cached dataset(s) OK")
        return EXIT_ERROR if mismatches else EXIT_OK

    def cmd_status(self, args: argparse.Namespace) -> int:
        """Print the store location, its size, and how much is materialized.

        Args:
            args: Parsed arguments; unused.

        Returns:
            The process exit code.
        """
        config = self.service.config
        root = config.store_path()
        registered = self.service.list_datasets()
        cached = [n for n in registered if config.processed_path(n).is_file()]
        implemented = [n for n in registered if self.service.preprocessor.supports(n)]

        print(f"store path   : {root}")
        print(f"exists       : {root.is_dir()}")
        print(f"raw size     : {self._human(self._tree_size(root / 'raw'))}")
        print(f"processed    : {self._human(self._tree_size(config.processed_dir()))}")
        print(f"registered   : {len(registered)} dataset(s)")
        print(f"implemented  : {len(implemented)} dataset(s) with a preprocessing routine")
        print(f"cached       : {len(cached)} dataset(s) ready to load")
        print(f"manifest     : {config.manifest_path()}")
        return EXIT_OK

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _tree_size(directory: Path) -> int:
        """Return the total size in bytes of every file under ``directory``.

        Args:
            directory: Directory to measure; may not exist.

        Returns:
            Total size in bytes, ``0`` when the directory is absent.
        """
        if not directory.is_dir():
            return 0
        return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())

    @staticmethod
    def _human(n_bytes: int) -> str:
        """Format a byte count as a human-readable string.

        Args:
            n_bytes: Size in bytes.

        Returns:
            The size rendered with a binary unit suffix (e.g. ``'1.4 GiB'``).
        """
        size = float(n_bytes)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if size < 1024.0 or unit == "TiB":
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TiB"


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point declared in ``pyproject.toml``.

    Args:
        argv: Argument vector, excluding the program name.

    Returns:
        The process exit code.
    """
    return CLI().run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
