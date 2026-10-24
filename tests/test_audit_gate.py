import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("audit_gate", Path("scripts/audit_gate.py"))
audit_gate = importlib.util.module_from_spec(_SPEC)
sys.modules["audit_gate"] = audit_gate
_SPEC.loader.exec_module(audit_gate)

ALLOWLIST = """
[[vuln]]
id = "PYSEC-2026-311"
aliases = ["CVE-2026-45829", "GHSA-f4j7-r4q5-qw2c"]
package = "chromadb"
version = "{version}"
reason = "No release has the fix yet."
"""

# The shape of the real record: a range that ends at last_affected, no fixed event.
OSV_LAST_AFFECTED = {
    "id": "PYSEC-2026-311",
    "affected": [
        {
            "package": {"ecosystem": "PyPI", "name": "chromadb"},
            "ranges": [
                {
                    "type": "ECOSYSTEM",
                    "events": [{"introduced": "1.0.0"}, {"last_affected": "1.5.9"}],
                }
            ],
            "versions": ["1.0.0", "1.5.8", "1.5.9"],
        }
    ],
}

OSV_FIXED = {
    "id": "PYSEC-2026-311",
    "affected": [
        {
            "package": {"ecosystem": "PyPI", "name": "chromadb"},
            "ranges": [
                {"type": "ECOSYSTEM", "events": [{"introduced": "1.0.0"}, {"fixed": "1.5.10"}]}
            ],
        }
    ],
}


def _report(version, vulns=()):
    return {
        "dependencies": [
            {"name": "chromadb", "version": version, "vulns": list(vulns)},
            {"name": "requests", "version": "2.33.1", "vulns": []},
        ],
        "fixes": [],
    }


def _vuln(vuln_id="PYSEC-2026-311", fix_versions=(), aliases=("CVE-2026-45829",)):
    return {"id": vuln_id, "fix_versions": list(fix_versions), "aliases": list(aliases)}


@pytest.fixture
def gate(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    def run(*, allowlist_version="1.5.9", report=None, osv=None, pypi_latest=None, args=()):
        allowlist = tmp_path / "allowlist.toml"
        allowlist.write_text(ALLOWLIST.format(version=allowlist_version))
        fetched = []

        reports = report if isinstance(report, dict) and "dependencies" not in report else None

        def fake_pip_audit(_cmd, requirements):
            if report is None:
                raise audit_gate.GateError(f"pip-audit gave no report for {requirements}")
            return reports[requirements] if reports else report

        def fake_fetch_json(url):
            fetched.append(url)
            if "osv.dev" in url and osv is not None:
                return osv
            if "pypi.org" in url and pypi_latest is not None:
                return {"info": {"version": pypi_latest}}
            raise audit_gate.GateError(f"could not read {url}")

        monkeypatch.setattr(audit_gate, "run_pip_audit", fake_pip_audit)
        monkeypatch.setattr(audit_gate, "fetch_json", fake_fetch_json)
        argv = ["--allowlist", str(allowlist), *args]
        if "--check-releases" not in args:
            for requirements in reports or ["requirements-api.txt"]:
                argv += ["-r", requirements]
        code = audit_gate.main(argv)
        return code, capsys.readouterr().out, fetched

    return run


def test_allowlisted_advisory_passes_without_network(gate):
    code, out, fetched = gate(report=_report("1.5.9", [_vuln()]))

    assert code == 0
    assert "1 matched" in out
    assert fetched == []


def test_advisory_missing_from_allowlist_fails(gate):
    code, out, _ = gate(report=_report("1.5.9", [_vuln(), _vuln("PYSEC-2026-9999", aliases=())]))

    assert code == 1
    assert "PYSEC-2026-9999" in out
    assert "not in the allowlist" in out


def test_entry_matches_through_an_alias(gate):
    vuln = _vuln("GHSA-f4j7-r4q5-qw2c", aliases=("CVE-2026-45829",))

    code, _, _ = gate(report=_report("1.5.9", [vuln]))

    assert code == 0


def test_entry_does_not_cover_another_version(gate):
    code, out, _ = gate(report=_report("1.5.8", [_vuln()]), osv=OSV_LAST_AFFECTED)

    assert code == 1
    assert "chromadb==1.5.8 has PYSEC-2026-311" in out
    assert "no requirements file pins chromadb==1.5.9 any more" in out
    assert "UNCONFIRMED" not in out


def test_fixed_version_on_an_accepted_advisory_fails(gate):
    code, out, _ = gate(report=_report("1.5.9", [_vuln(fix_versions=["1.5.10"])]))

    assert code == 1
    assert "fixed in 1.5.10. Bump chromadb, then delete the entry." in out


def test_bump_past_last_affected_is_unconfirmed(gate):
    code, out, _ = gate(report=_report("1.5.10"), osv=OSV_LAST_AFFECTED)

    assert code == 1
    assert "UNCONFIRMED PYSEC-2026-311: chromadb 1.5.10 is pinned" in out
    assert "release notes for CVE-2026-45829" in out
    assert 'set the version of entry #1 to "1.5.10"' in out
    assert out.count("ERROR") == 1


def test_bump_to_a_release_osv_marks_fixed_asks_for_deletion(gate):
    code, out, _ = gate(report=_report("1.5.10"), osv=OSV_FIXED)

    assert code == 1
    assert "no requirements file pins chromadb==1.5.9 any more. Delete the entry" in out
    assert "UNCONFIRMED" not in out


def test_entry_kept_by_hand_past_last_affected_passes_with_notice(gate):
    code, out, _ = gate(allowlist_version="1.5.10", report=_report("1.5.10"), osv=OSV_LAST_AFFECTED)

    assert code == 0
    assert "NOTICE: entry #1 (PYSEC-2026-311, chromadb==1.5.10) is kept by hand" in out


def test_entry_kept_by_hand_fails_once_osv_records_the_fix(gate):
    code, out, _ = gate(allowlist_version="1.5.10", report=_report("1.5.10"), osv=OSV_FIXED)

    assert code == 1
    assert "OSV says PYSEC-2026-311 is fixed in 1.5.10. Delete the entry." in out


def test_bump_in_only_one_requirements_file_fails(gate):
    reports = {
        "requirements-api.txt": _report("1.5.10"),
        "requirements-ui.txt": _report("1.5.9", [_vuln()]),
    }

    code, out, _ = gate(report=reports, osv=OSV_LAST_AFFECTED)

    assert code == 1
    assert "UNCONFIRMED PYSEC-2026-311: chromadb 1.5.10 is pinned" in out
    assert "pin one version everywhere first" in out


def test_entry_kept_by_hand_does_not_cover_a_newer_pin_elsewhere(gate):
    reports = {
        "requirements-api.txt": _report("1.5.10"),
        "requirements-ui.txt": _report("1.5.11"),
    }

    code, out, _ = gate(allowlist_version="1.5.10", report=reports, osv=OSV_LAST_AFFECTED)

    assert code == 1
    assert "chromadb 1.5.11 is pinned" in out


def test_unreadable_osv_record_fails(gate):
    code, out, _ = gate(report=_report("1.5.10"))

    assert code == 1
    assert "OSV record could not be checked" in out


def test_package_no_longer_audited_fails(gate):
    report = {"dependencies": [{"name": "requests", "version": "2.33.1", "vulns": []}]}

    code, out, _ = gate(report=report)

    assert code == 1
    assert "chromadb is not in the audited requirements" in out


def test_failed_audit_fails_without_judging_entries(gate):
    code, out, fetched = gate(report=None)

    assert code == 1
    assert "audit failed" in out
    assert fetched == []


def test_newer_release_on_pypi_fails_release_check(gate):
    code, out, _ = gate(pypi_latest="1.5.10", args=["--check-releases"])

    assert code == 1
    assert "chromadb 1.5.10 is out and allowlist entries #1 still cover 1.5.9" in out


def test_release_check_passes_when_allowlist_is_current(gate):
    code, _, _ = gate(pypi_latest="1.5.9", args=["--check-releases"])

    assert code == 0


def test_unreadable_pypi_fails_release_check(gate):
    code, out, _ = gate(args=["--check-releases"])

    assert code == 1
    assert "could not read the latest chromadb release" in out


@pytest.mark.parametrize(
    ("version", "expected"),
    [("0.9.0", "unaffected"), ("1.0.0", "affected"), ("1.5.9", "affected"), ("1.5.10", "unknown")],
)
def test_osv_status_with_last_affected(version, expected):
    assert audit_gate.osv_status(OSV_LAST_AFFECTED, "chromadb", version) == expected


@pytest.mark.parametrize(("version", "expected"), [("1.5.9", "affected"), ("1.5.10", "fixed")])
def test_osv_status_with_fixed_event(version, expected):
    assert audit_gate.osv_status(OSV_FIXED, "chromadb", version) == expected


@pytest.mark.parametrize(
    ("version", "expected"), [("0.3.0", "affected"), ("0.5.0", "fixed"), ("1.6.0", "unknown")]
)
def test_osv_status_lets_an_open_question_outrank_an_older_fix(version, expected):
    record = {
        "affected": [
            {
                "package": {"ecosystem": "PyPI", "name": "chromadb"},
                "ranges": [
                    {"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "0.4.0"}]},
                    {
                        "type": "ECOSYSTEM",
                        "events": [{"introduced": "1.0.0"}, {"last_affected": "1.5.9"}],
                    },
                ],
            }
        ]
    }

    assert audit_gate.osv_status(record, "chromadb", version) == expected


def test_osv_status_reports_withdrawn_records():
    record = {**OSV_FIXED, "withdrawn": "2026-10-01T00:00:00Z"}

    assert audit_gate.osv_status(record, "chromadb", "1.5.9") == "withdrawn"


def test_run_pip_audit_rejects_output_that_is_not_a_report():
    with pytest.raises(audit_gate.GateError, match="no report"):
        audit_gate.run_pip_audit([sys.executable, "-c", "print('oops')"], "requirements-api.txt")


def test_run_pip_audit_rejects_unexpected_exit_codes():
    script = "import sys; print('{\"dependencies\": []}'); sys.exit(2)"

    with pytest.raises(audit_gate.GateError, match="exit 2"):
        audit_gate.run_pip_audit([sys.executable, "-c", script], "requirements-api.txt")


@pytest.mark.parametrize(
    "body",
    [
        'id = "X"\npackage = "chromadb"\nversion = "1.5.9"\n',
        'id = "X"\npackage = "chromadb"\nversion = "1.5.9"\nreason = " "\n',
        'id = "X"\npackage = "chromadb"\nversion = "1.5.9"\nreason = "r"\nexpires = 2027-01-01\n',
        'id = "X"\npackage = "chromadb"\nversion = "not a version"\nreason = "r"\n',
    ],
)
def test_load_allowlist_rejects_bad_entries(tmp_path, body):
    path = tmp_path / "allowlist.toml"
    path.write_text("[[vuln]]\n" + body)

    with pytest.raises(audit_gate.GateError):
        audit_gate.load_allowlist(str(path))


def test_repository_allowlist_loads():
    entries = audit_gate.load_allowlist(".github/pip-audit-allowlist.toml")

    assert entries
    assert all(entry.reason for entry in entries)
