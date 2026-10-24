#!/usr/bin/env python3
"""Run pip-audit and accept only the advisories in a version-pinned allowlist.

pip-audit runs with no --ignore-vuln flags. Each allowlist entry accepts one
advisory for one exact package version. The run fails when:

- pip-audit reports an advisory that no entry covers;
- an accepted advisory gets a fixed version;
- an entry covers a version the requirements no longer pin, for example after
  a bump;
- a pinned version of an allowlisted package is not reported, has no entry,
  and OSV does not say it is fixed. Advisories that end at ``last_affected``
  (the ChromaDB ones do) stop being reported for any later release, fixed or
  not, so such a release has to be checked by hand ("UNCONFIRMED");
- pip-audit, OSV or PyPI could not be read.

With ``--check-releases`` it also fails when PyPI has a release newer than an
entry's version. CI runs that weekly, because an advisory that ends at
``last_affected`` never gets the fixed version that would show a fix shipped.

Usage::

    python scripts/audit_gate.py --allowlist .github/pip-audit-allowlist.toml \\
        -r requirements-api.txt -r requirements-ui.txt
    python scripts/audit_gate.py --allowlist .github/pip-audit-allowlist.toml \\
        --check-releases
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from packaging.version import InvalidVersion, Version

OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{id}"
PYPI_PROJECT_URL = "https://pypi.org/pypi/{package}/json"
REQUIRED_KEYS = {"id", "package", "version", "reason"}
OPTIONAL_KEYS = {"aliases"}


class GateError(Exception):
    """Something the gate depends on could not be read, so it cannot pass."""


def canonical_name(name: str) -> str:
    """PEP 503 project name, the form pip-audit prints."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass
class Entry:
    number: int
    id: str
    aliases: frozenset[str]
    package: str
    version: str
    reason: str
    matched_in: list[str] = field(default_factory=list)

    @property
    def ids(self) -> frozenset[str]:
        return self.aliases | {self.id}

    @property
    def cves(self) -> str:
        return ", ".join(sorted(i for i in self.ids if i.startswith("CVE-"))) or self.id

    def label(self) -> str:
        return f"entry #{self.number} ({self.id}, {self.package}=={self.version})"


def load_allowlist(path: str) -> list[Entry]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    if set(data) - {"vuln"}:
        raise GateError(f"{path}: unknown top-level keys {sorted(set(data) - {'vuln'})}")

    entries = []
    for number, raw in enumerate(data.get("vuln", []), start=1):
        if REQUIRED_KEYS - set(raw):
            raise GateError(f"{path}: entry #{number} is missing {sorted(REQUIRED_KEYS - set(raw))}")
        if set(raw) - REQUIRED_KEYS - OPTIONAL_KEYS:
            unknown = sorted(set(raw) - REQUIRED_KEYS - OPTIONAL_KEYS)
            raise GateError(f"{path}: entry #{number} has unknown keys {unknown}")
        if not str(raw["reason"]).strip():
            raise GateError(f"{path}: entry #{number} needs a reason")
        try:
            Version(str(raw["version"]))
        except InvalidVersion:
            raise GateError(f"{path}: entry #{number} has an invalid version") from None
        entries.append(
            Entry(
                number=number,
                id=str(raw["id"]),
                aliases=frozenset(str(alias) for alias in raw.get("aliases", [])),
                package=canonical_name(str(raw["package"])),
                version=str(raw["version"]),
                reason=str(raw["reason"]),
            )
        )
    return entries


def run_pip_audit(pip_audit: list[str], requirements: str) -> dict:
    cmd = [*pip_audit, "-r", requirements, "--strict", "--format", "json"]
    cmd += ["--progress-spinner", "off"]
    print(f"$ {shlex.join(cmd)}", file=sys.stderr, flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    # Exit 1 means either "vulnerabilities found" (report on stdout) or a
    # failure (no report). Only the JSON tells the two apart.
    try:
        report = json.loads(proc.stdout) if proc.returncode in (0, 1) else None
    except json.JSONDecodeError:
        report = None
    if not isinstance(report, dict) or not isinstance(report.get("dependencies"), list):
        raise GateError(f"pip-audit gave no report for {requirements} (exit {proc.returncode})")
    return report


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "video-search-engine-audit-gate"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise GateError(f"could not read {url}: {exc}") from None


def _pypi_ranges(record: dict, package: str):
    for affected in record.get("affected", []):
        pkg = affected.get("package", {})
        if pkg.get("ecosystem") == "PyPI" and canonical_name(pkg.get("name", "")) == package:
            yield affected


def osv_status(record: dict, package: str, version: str) -> str:
    """Say what an OSV record means for one version.

    Returns "withdrawn", "affected", "unknown" (past ``last_affected``: no
    longer listed as affected, but not marked fixed), "fixed" or "unaffected"
    (before the first ``introduced``). When ranges disagree, the least
    reassuring answer wins.
    """
    if record.get("withdrawn"):
        return "withdrawn"
    target = Version(version)
    statuses = set()
    for affected in _pypi_ranges(record, package):
        if version in affected.get("versions", []):
            statuses.add("affected")
        for rng in affected.get("ranges", []):
            if rng.get("type") != "ECOSYSTEM":
                continue
            events = [item for event in rng.get("events", []) for item in event.items()]
            status = "unaffected"
            for kind, value in sorted(events, key=lambda item: Version(item[1])):
                if Version(value) > target:
                    break
                if kind == "introduced":
                    status = "affected"
                elif kind == "fixed":
                    status = "fixed"
                elif kind == "last_affected":
                    status = "affected" if Version(value) == target else "unknown"
            statuses.add(status)
    for status in ("affected", "unknown", "fixed"):
        if status in statuses:
            return status
    return "unaffected"


def osv_fixed_versions(record: dict, package: str, version: str) -> str:
    """The fixed events at or below ``version``."""
    fixed = {
        event["fixed"]
        for affected in _pypi_ranges(record, package)
        for rng in affected.get("ranges", [])
        for event in rng.get("events", [])
        if "fixed" in event and Version(event["fixed"]) <= Version(version)
    }
    return ", ".join(sorted(fixed, key=Version))


def check_reports(
    reports: list[tuple[str, dict]], entries: list[Entry], errors: list[str]
) -> tuple[dict[str, set[str]], set[tuple[str, str, str]]]:
    """Match each finding to an entry.

    Return the audited versions per package, and every (package, version,
    advisory id or alias) that pip-audit reported.
    """
    audited: dict[str, set[str]] = {}
    reported: set[tuple[str, str, str]] = set()
    for label, report in reports:
        for dep in report["dependencies"]:
            name = canonical_name(dep["name"])
            if "skip_reason" in dep:
                errors.append(f"{label}: {name} was not audited: {dep['skip_reason']}")
                continue
            version = dep["version"]
            audited.setdefault(name, set()).add(version)
            for vuln in dep.get("vulns", []):
                ids = {vuln["id"], *vuln.get("aliases", [])}
                reported.update((name, version, i) for i in ids)
                fixes = ", ".join(vuln.get("fix_versions", []))
                entry = next(
                    (
                        e
                        for e in entries
                        if e.package == name and e.version == version and e.ids & ids
                    ),
                    None,
                )
                if entry is None:
                    aliases = ", ".join(sorted(ids - {vuln["id"]})) or "no aliases"
                    errors.append(
                        f"{label}: {name}=={version} has {vuln['id']} ({aliases}),"
                        f" {f'fixed in {fixes}' if fixes else 'no fixed version'}."
                        " It is not in the allowlist."
                    )
                    continue
                entry.matched_in.append(label)
                if fixes:
                    errors.append(
                        f"{entry.label()}: fixed in {fixes}."
                        f" Bump {name}, then delete the entry."
                    )
    return audited, reported


def check_advisory(
    group: list[Entry],
    pinned: set[str],
    silent: list[str],
    record: dict,
    errors: list[str],
    notices: list[str],
) -> None:
    """Account for every pinned version that pip-audit says nothing about."""
    first = group[0]
    package, vuln_id = first.package, first.id
    covered = {entry.version: entry for entry in group}
    stale = [entry for entry in group if entry.version not in pinned]
    numbers = ", ".join(f"#{entry.number}" for entry in stale)
    unresolved = False

    for version in silent:
        status = osv_status(record, package, version)
        not_affected = {
            "withdrawn": "is withdrawn",
            "fixed": f"is fixed in {osv_fixed_versions(record, package, version)}",
            "unaffected": f"does not affect {version}",
        }.get(status)
        entry = covered.get(version)
        if entry is not None:
            # Kept by hand for a release past last_affected, or the advisory changed.
            if not_affected:
                errors.append(f"{entry.label()}: OSV says {vuln_id} {not_affected}. Delete the entry.")
            else:
                notices.append(
                    f"{entry.label()} is kept by hand: pip-audit does not report {vuln_id}"
                    f" for {version} and OSV records no fix."
                )
            continue
        if not_affected:
            continue

        unresolved = True
        if status == "affected":
            finding = (
                f"{package} {version} is pinned and OSV lists it as affected by {vuln_id},"
                " but pip-audit does not report it."
            )
        else:
            finding = (
                f"UNCONFIRMED {vuln_id}: {package} {version} is pinned and pip-audit does"
                " not report the advisory for it, but OSV records no fix."
            )
        if len(pinned) > 1:
            pins = ", ".join(sorted(pinned, key=Version))
            action = f"The requirements pin {package} {pins}; pin one version everywhere first."
        elif status == "affected":
            action = f'Set the version of entry {numbers} to "{version}".'
        else:
            action = (
                f"Check the {package} {version} release notes for {first.cves}. If {version}"
                f" fixes it, delete entry {numbers} and cite the release note in the commit"
                f' message. If not, set the version of entry {numbers} to "{version}".'
            )
        errors.append(f"{finding} {action}")

    if unresolved:
        return
    for entry in stale:
        errors.append(
            f"{entry.label()}: no requirements file pins {package}=={entry.version} any"
            " more. Delete the entry, or set its version to the one now pinned if the"
            " advisory still applies."
        )


def check_entries(
    entries: list[Entry],
    audited: dict[str, set[str]],
    reported: set[tuple[str, str, str]],
    errors: list[str],
    notices: list[str],
) -> None:
    """Check each allowlisted advisory against every version the requirements pin."""
    groups: dict[tuple[str, str], list[Entry]] = {}
    for entry in entries:
        groups.setdefault((entry.package, entry.id), []).append(entry)

    for (package, vuln_id), group in groups.items():
        pinned = audited.get(package, set())
        if not pinned:
            for entry in group:
                errors.append(
                    f"{entry.label()}: {package} is not in the audited requirements."
                    " Delete the entry."
                )
            continue
        ids = frozenset().union(*(entry.ids for entry in group))
        # pip-audit stops reporting an advisory past last_affected whether or
        # not the release has the fix, so its silence has to be checked.
        silent = sorted(
            (v for v in pinned if not any((package, v, i) in reported for i in ids)),
            key=Version,
        )
        if not silent:
            check_advisory(group, pinned, silent, {}, errors, notices)
            continue
        try:
            record = fetch_json(OSV_VULN_URL.format(id=vuln_id))
            check_advisory(group, pinned, silent, record, errors, notices)
        except (GateError, InvalidVersion) as exc:
            errors.append(
                f"{vuln_id}: pip-audit does not report it for {package}"
                f" {', '.join(silent)} and the OSV record could not be checked: {exc!r}"
            )


def check_releases(entries: list[Entry], errors: list[str]) -> None:
    for package in sorted({entry.package for entry in entries}):
        try:
            latest = fetch_json(PYPI_PROJECT_URL.format(package=package))["info"]["version"]
            latest_version = Version(latest)
        except (GateError, KeyError, TypeError, InvalidVersion) as exc:
            errors.append(f"could not read the latest {package} release from PyPI: {exc!r}")
            continue
        behind = [
            entry
            for entry in entries
            if entry.package == package and Version(entry.version) < latest_version
        ]
        if behind:
            numbers = ", ".join(f"#{entry.number}" for entry in behind)
            versions = ", ".join(sorted({entry.version for entry in behind}, key=Version))
            errors.append(
                f"{package} {latest} is out and allowlist entries {numbers} still cover"
                f" {versions}. Bump {package} (Dependabot should open a PR) and check"
                f" whether {latest} fixes them."
            )


def annotate(kind: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{kind}::{message}")
    else:
        print(f"{kind.upper()}: {message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allowlist", required=True, help="TOML allowlist of accepted advisories")
    parser.add_argument(
        "-r",
        "--requirement",
        action="append",
        default=[],
        help="requirements file to audit; repeat it so every file is read in one run",
    )
    parser.add_argument("--pip-audit", default="pip-audit", help="pip-audit command to run")
    parser.add_argument(
        "--check-releases",
        action="store_true",
        help="fail when PyPI has a release newer than an entry's version",
    )
    args = parser.parse_args(argv)
    if not args.requirement and not args.check_releases:
        parser.error("give at least one -r or --check-releases")

    try:
        entries = load_allowlist(args.allowlist)
    except (GateError, OSError, tomllib.TOMLDecodeError) as exc:
        annotate("error", f"could not load the allowlist: {exc}")
        return 1

    errors: list[str] = []
    notices: list[str] = []
    if args.requirement:
        try:
            pip_audit = shlex.split(args.pip_audit)
            reports = [(req, run_pip_audit(pip_audit, req)) for req in args.requirement]
        except GateError as exc:
            errors.append(f"audit failed: {exc}")
        else:
            # An entry that matched nothing only means something once every
            # file has been read, so all of them go through one run.
            audited, reported = check_reports(reports, entries, errors)
            check_entries(entries, audited, reported, errors, notices)
    if args.check_releases:
        check_releases(entries, errors)

    for message in notices:
        annotate("notice", message)
    for message in errors:
        annotate("error", message)
    matched = sum(1 for entry in entries if entry.matched_in)
    print(
        f"audit_gate: {len(entries)} allowlist entries, {matched} matched,"
        f" {len(notices)} kept by hand, {len(errors)} error(s)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
