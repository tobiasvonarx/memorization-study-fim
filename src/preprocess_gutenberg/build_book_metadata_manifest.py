#!/usr/bin/env python3
"""Build a Gutenberg book-metadata manifest keyed by source_book_id.

The cached `manu/project_gutenberg` dataset used in preprocessing only exposes
`id` and `text`. In practice, the `source_book_id` propagated through the
preprocessing pipeline matches that dataset `id`, which appears to be of the
form `<ebook_id>-<suffix>`. The numeric prefix before the first dash matches
the Project Gutenberg ebook number, which can be joined against the official
Project Gutenberg catalog metadata.

This script:
1. scans Gutenberg preprocessing manifests for unique `source_book_id` values
2. extracts the numeric Project Gutenberg ebook id from each source id
3. downloads (or reuses) the official Project Gutenberg catalog CSV
4. writes a compact JSONL manifest keyed by `source_book_id`

The resulting JSONL can be passed to the notebook via `GUTENBERG_METADATA_PATH`
or placed next to the filtered outputs as `book_metadata.jsonl`.
"""

import argparse
import csv
import gzip
import json
import os
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
SOURCE_BOOK_ID_COLUMN = "source_book_id"
DEFAULT_MANIFEST_NAMES = (
    "scores.jsonl",
    "thresholded_text.jsonl",
    "text.jsonl",
    "semantic_dedup_removed_text.jsonl",
)
SOURCE_ID_PATTERN = re.compile(r"^(?P<ebook_id>\d+)(?:$|[-_].*)")
SOURCE_ID_ALIASES = {
    # `manu/project_gutenberg` contains this non-numeric id for PG ebook 10453:
    # A Practical Physiology: A Text-Book for Higher Schools.
    "phys-0": "10453",
}
DATE_SPAN_PATTERN = re.compile(r",?\s*\b\d{3,4}\s*-\s*\d{0,4}\b")
ROLE_PATTERN = re.compile(r"\[([^\]]+)\]\s*$")
TRANSLITERATION_MAP = str.maketrans(
    {
        "Æ": "AE",
        "æ": "ae",
        "Ð": "D",
        "ð": "d",
        "Ł": "L",
        "ł": "l",
        "Ø": "O",
        "ø": "o",
        "Þ": "Th",
        "þ": "th",
        "ß": "ss",
    }
)


def find_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config.env").exists():
            return candidate
    raise FileNotFoundError("Could not locate config.env from the current working directory.")


def parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    pattern = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)="?(.*?)"?$')
    variable_pattern = re.compile(r"\$\{([^}]+)\}")

    def resolve_variable(expression: str) -> str:
        if ":-" in expression:
            name, default = expression.split(":-", 1)
            value = values.get(name, os.environ.get(name))
            return value if value not in (None, "") else default
        return values.get(expression, os.environ.get(expression, "${" + expression + "}"))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        key, raw_value = match.group(1), match.group(2)
        values[key] = variable_pattern.sub(lambda found: resolve_variable(found.group(1)), raw_value)
    return values


def resolve_filtered_dir(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()

    repo_root = find_repo_root()
    config = parse_env_file(repo_root / "config.env")
    env_value = os.getenv("GUTENBERG_FILTERED_DIR") or config.get("GUTENBERG_FILTERED_DIR")
    if not env_value:
        raise ValueError("Could not resolve GUTENBERG_FILTERED_DIR from CLI, env, or config.env.")
    return Path(env_value).expanduser().resolve()


def resolve_default_output_path(filtered_dir: Path, cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    return filtered_dir / "book_metadata.jsonl"


def resolve_default_catalog_cache_path(filtered_dir: Path, cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    return filtered_dir / "pg_catalog.csv.gz"


def load_source_book_ids(filtered_dir: Path, manifest_names: Iterable[str]) -> Set[str]:
    source_ids: Set[str] = set()
    for manifest_name in manifest_names:
        path = filtered_dir / manifest_name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                source_book_id = row.get(SOURCE_BOOK_ID_COLUMN)
                if source_book_id:
                    source_ids.add(str(source_book_id))
    return source_ids


def extract_ebook_id(source_book_id: str) -> Optional[str]:
    if source_book_id in SOURCE_ID_ALIASES:
        return SOURCE_ID_ALIASES[source_book_id]
    match = SOURCE_ID_PATTERN.match(source_book_id)
    if not match:
        return None
    return match.group("ebook_id")


def normalize_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).translate(TRANSLITERATION_MAP)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = DATE_SPAN_PATTERN.sub(" ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.casefold())
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def split_semicolon_terms(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def parse_contributors(author_value: Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    contributors: List[Dict[str, Any]] = []
    for raw_part in split_semicolon_terms(author_value):
        roles: List[str] = []
        name = raw_part.strip()
        role_match = ROLE_PATTERN.search(name)
        if role_match:
            roles = [role.strip() for role in role_match.group(1).split(",") if role.strip()]
            name = ROLE_PATTERN.sub("", name).strip()

        display_name = name
        name_without_dates = DATE_SPAN_PATTERN.sub("", name).strip(" ,")
        contributors.append(
            {
                "name": display_name,
                "name_without_dates": name_without_dates or display_name,
                "name_key": normalize_key(name_without_dates or display_name),
                "roles": roles,
                "raw": raw_part,
            }
        )

    primary_author = None
    for contributor in contributors:
        role_keys = {normalize_key(role) for role in contributor["roles"]}
        if not role_keys.intersection({"translator", "editor", "illustrator", "commentator"}):
            primary_author = contributor["name_without_dates"]
            break
    if primary_author is None and contributors:
        primary_author = contributors[0]["name_without_dates"]
    return contributors, primary_author


def build_work_key(primary_author: Optional[str], title: Optional[str]) -> Optional[str]:
    author_key = normalize_key(primary_author)
    title_key = normalize_key(title)
    if not author_key or not title_key:
        return None
    return f"{author_key} | {title_key}"


def enrich_catalog_row(row: Dict[str, Any]) -> Dict[str, Any]:
    contributors, primary_author = parse_contributors(row.get("author"))
    contributor_names = [contributor["name_without_dates"] for contributor in contributors]
    contributor_roles = sorted({role for contributor in contributors for role in contributor["roles"]})
    issued = row.get("issued")
    issued_year = None
    if issued:
        year_match = re.match(r"^(\d{4})", str(issued))
        if year_match:
            issued_year = int(year_match.group(1))

    enriched = dict(row)
    enriched.update(
        {
            "title_key": normalize_key(row.get("title")),
            "author_key": normalize_key(row.get("author")),
            "primary_author": primary_author,
            "primary_author_key": normalize_key(primary_author),
            "contributor_names": contributor_names,
            "contributor_roles": contributor_roles,
            "contributor_keys": [contributor["name_key"] for contributor in contributors if contributor["name_key"]],
            "language_codes": split_semicolon_terms(row.get("language")),
            "subject_terms": split_semicolon_terms(row.get("subjects")),
            "locc_classes": split_semicolon_terms(row.get("locc")),
            "bookshelf_terms": split_semicolon_terms(row.get("bookshelves")),
            "issued_year": issued_year,
            "work_key": build_work_key(primary_author, row.get("title")),
        }
    )
    return enriched


def download_catalog_if_needed(url: str, cache_path: Path, force_download: bool) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force_download:
        return cache_path
    print(f"Downloading Project Gutenberg catalog from {url} -> {cache_path}", flush=True)
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def read_catalog_rows(catalog_path: Path) -> Dict[str, Dict[str, Any]]:
    with gzip.open(catalog_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: Dict[str, Dict[str, Any]] = {}
        for row in reader:
            ebook_id = (row.get("Text#") or "").strip()
            if not ebook_id:
                continue
            rows[ebook_id] = enrich_catalog_row({
                "gutenberg_ebook_id": ebook_id,
                "type": (row.get("Type") or "").strip() or None,
                "issued": (row.get("Issued") or "").strip() or None,
                "title": (row.get("Title") or "").strip() or None,
                "language": (row.get("Language") or "").strip() or None,
                "author": (row.get("Authors") or "").strip() or None,
                "subjects": (row.get("Subjects") or "").strip() or None,
                "locc": (row.get("LoCC") or "").strip() or None,
                "bookshelves": (row.get("Bookshelves") or "").strip() or None,
            })
        return rows


def build_manifest_rows(source_ids: Iterable[str], catalog_rows: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    manifest_rows: List[Dict[str, Any]] = []
    unresolved: List[str] = []

    for source_book_id in sorted(source_ids):
        ebook_id = extract_ebook_id(source_book_id)
        if ebook_id is None:
            unresolved.append(source_book_id)
            continue

        catalog_row = catalog_rows.get(ebook_id)
        if catalog_row is None:
            unresolved.append(source_book_id)
            continue

        row = {"source_book_id": source_book_id, **catalog_row}
        manifest_rows.append(row)

    return manifest_rows, unresolved


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filtered-dir",
        type=str,
        default=None,
        help="Filtered Gutenberg output directory containing scores/text manifests. Defaults to $GUTENBERG_FILTERED_DIR.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output JSONL manifest path. Defaults to <filtered-dir>/book_metadata.jsonl.",
    )
    parser.add_argument(
        "--catalog-url",
        type=str,
        default=CATALOG_URL,
        help="Project Gutenberg catalog CSV URL.",
    )
    parser.add_argument(
        "--catalog-cache-path",
        type=str,
        default=None,
        help="Where to cache pg_catalog.csv.gz. Defaults to <filtered-dir>/pg_catalog.csv.gz.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the catalog even if a cached copy exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    filtered_dir = resolve_filtered_dir(args.filtered_dir)
    if not filtered_dir.exists():
        raise FileNotFoundError(f"Filtered output directory does not exist: {filtered_dir}")

    output_path = resolve_default_output_path(filtered_dir, args.output_path)
    catalog_cache_path = resolve_default_catalog_cache_path(filtered_dir, args.catalog_cache_path)

    source_ids = load_source_book_ids(filtered_dir, DEFAULT_MANIFEST_NAMES)
    if not source_ids:
        raise RuntimeError(f"No source_book_id values found under {filtered_dir}")

    parsed_ids = [source_book_id for source_book_id in source_ids if extract_ebook_id(source_book_id) is not None]
    print(
        "Found {:,} unique source_book_id values ({:,} match the expected <ebook_id>-suffix pattern).".format(
            len(source_ids),
            len(parsed_ids),
        ),
        flush=True,
    )

    catalog_path = download_catalog_if_needed(args.catalog_url, catalog_cache_path, args.force_download)
    catalog_rows = read_catalog_rows(catalog_path)
    print("Loaded {:,} catalog rows from {}".format(len(catalog_rows), catalog_path), flush=True)

    manifest_rows, unresolved = build_manifest_rows(source_ids, catalog_rows)
    write_jsonl(output_path, manifest_rows)

    print(f"Wrote {len(manifest_rows):,} metadata rows to {output_path}", flush=True)
    unresolved_path = output_path.with_suffix(output_path.suffix + ".unresolved.txt")
    if unresolved:
        unresolved_preview = ", ".join(unresolved[:10])
        print(
            "Warning: could not resolve {:,} source ids. First few: {}".format(
                len(unresolved),
                unresolved_preview,
            ),
            flush=True,
        )
        unresolved_path.write_text("\n".join(unresolved) + "\n", encoding="utf-8")
        print(f"Wrote unresolved source ids to {unresolved_path}", flush=True)
    elif unresolved_path.exists():
        unresolved_path.unlink()
        print(f"Removed stale unresolved source id file {unresolved_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
