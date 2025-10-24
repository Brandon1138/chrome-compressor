"""Extractors for various corpus formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import json
import subprocess
import zipfile
import gzip
import tarfile

from reducelang.alphabet import Alphabet


class CorpusExtractor(ABC):
    """Abstract base class for corpus extractors."""

    @abstractmethod
    def extract(self, raw_path: Path, output_path: Path, alphabet: Alphabet) -> dict[str, Any]:
        """Extract and normalize text from ``raw_path`` to ``output_path``."""


class PlainTextExtractor(CorpusExtractor):
    def extract(self, raw_path: Path, output_path: Path, alphabet: Alphabet) -> dict[str, Any]:
        # Read as UTF-8 text, normalize, and write out
        content = raw_path.read_text(encoding="utf-8", errors="ignore")
        norm = alphabet.normalize(content)
        with output_path.open("w", encoding="utf-8") as out:
            out.write(norm)
        return {
            "char_count": len(norm),
            "unique_chars": set(norm),
            "raw_size_bytes": raw_path.stat().st_size,
            "processed_size_bytes": output_path.stat().st_size,
        }


class WikipediaExtractor(CorpusExtractor):
    def extract(self, raw_path: Path, output_path: Path, alphabet: Alphabet) -> dict[str, Any]:
        temp_dir = output_path.parent / "_wikiextractor_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Try wikiextractor CLI first; fall back to python -m wikiextractor
            cmd_cli = [
                "wikiextractor",
                "--json",
                "-o",
                str(temp_dir),
                str(raw_path),
            ]
            try:
                subprocess.run(cmd_cli, check=True)
            except FileNotFoundError:
                cmd_module = [
                    "python",
                    "-m",
                    "wikiextractor",
                    "--json",
                    "-o",
                    str(temp_dir),
                    str(raw_path),
                ]
                try:
                    subprocess.run(cmd_module, check=True)
                except FileNotFoundError as e:
                    raise ImportError(
                        "wikiextractor not found. Install it (`pip install wikiextractor`) and ensure the CLI is on PATH, or run via `python -m wikiextractor`."
                    ) from e

            char_count = 0
            unique_chars: set[str] = set()
            with output_path.open("w", encoding="utf-8") as out:
                for json_file in sorted(temp_dir.rglob("*.json")):
                    with json_file.open("r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            text = obj.get("text", "")
                            norm = alphabet.normalize(text)
                            out.write(norm)
                            char_count += len(norm)
                            unique_chars.update(norm)

            return {
                "char_count": char_count,
                "unique_chars": unique_chars,
                "raw_size_bytes": raw_path.stat().st_size,
                "processed_size_bytes": output_path.stat().st_size,
            }
        finally:
            # Clean temporary files
            for p in sorted(temp_dir.rglob("*"), reverse=True):
                try:
                    if p.is_file():
                        p.unlink(missing_ok=True)
                    else:
                        p.rmdir()
                except Exception:
                    pass
            try:
                temp_dir.rmdir()
            except Exception:
                pass


class NLTKBrownExtractor(CorpusExtractor):
    def extract(self, raw_path: Path, output_path: Path, alphabet: Alphabet) -> dict[str, Any]:
        try:
            import nltk
            from nltk.corpus import brown
        except ImportError as e:
            raise ImportError("nltk is required for NLTKBrownExtractor. Install 'nltk'.") from e

        # Ensure corpus is present
        try:
            nltk.data.find("corpora/brown")
        except LookupError:
            nltk.download("brown", quiet=True)

        tokens = brown.words()
        text = " ".join(tokens)
        norm = alphabet.normalize(text)
        with output_path.open("w", encoding="utf-8") as out:
            out.write(norm)
        return {
            "char_count": len(norm),
            "unique_chars": set(norm),
            "raw_size_bytes": 0,
            "processed_size_bytes": output_path.stat().st_size,
        }


class NLTKGutenbergExtractor(CorpusExtractor):
    def extract(self, raw_path: Path, output_path: Path, alphabet: Alphabet) -> dict[str, Any]:
        try:
            import nltk
            from nltk.corpus import gutenberg
        except ImportError as e:
            raise ImportError("nltk is required for NLTKGutenbergExtractor. Install 'nltk'.") from e

        # Ensure corpus is present
        try:
            nltk.data.find("corpora/gutenberg")
        except LookupError:
            nltk.download("gutenberg", quiet=True)

        # Concatenate all gutenberg files' raw text
        texts: list[str] = []
        for fileid in gutenberg.fileids():
            try:
                texts.append(gutenberg.raw(fileid))
            except Exception:
                continue
        text = "\n".join(texts)
        norm = alphabet.normalize(text)
        with output_path.open("w", encoding="utf-8") as out:
            out.write(norm)
        return {
            "char_count": len(norm),
            "unique_chars": set(norm),
            "raw_size_bytes": 0,
            "processed_size_bytes": output_path.stat().st_size,
        }


class ZipTextExtractor(CorpusExtractor):
    def extract(self, raw_path: Path, output_path: Path, alphabet: Alphabet) -> dict[str, Any]:
        with zipfile.ZipFile(raw_path) as zf:
            texts: list[str] = []
            for info in zf.infolist():
                if info.is_dir():
                    continue
                with zf.open(info) as f:
                    data = f.read()
                    try:
                        s = data.decode("utf-8")
                    except UnicodeDecodeError:
                        s = data.decode("latin-1")
                    texts.append(s)
        content = "\n".join(texts)
        norm = alphabet.normalize(content)
        with output_path.open("w", encoding="utf-8") as out:
            out.write(norm)
        return {
            "char_count": len(norm),
            "unique_chars": set(norm),
            "raw_size_bytes": raw_path.stat().st_size,
            "processed_size_bytes": output_path.stat().st_size,
        }


class GzipTextExtractor(CorpusExtractor):
    def extract(self, raw_path: Path, output_path: Path, alphabet: Alphabet) -> dict[str, Any]:
        with gzip.open(raw_path, "rt", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        norm = alphabet.normalize(content)
        with output_path.open("w", encoding="utf-8") as out:
            out.write(norm)
        return {
            "char_count": len(norm),
            "unique_chars": set(norm),
            "raw_size_bytes": raw_path.stat().st_size,
            "processed_size_bytes": output_path.stat().st_size,
        }


class EuroparlExtractor(CorpusExtractor):
    def extract(self, raw_path: Path, output_path: Path, alphabet: Alphabet) -> dict[str, Any]:
        member_file = None
        with tarfile.open(raw_path, "r:gz") as tf:
            candidates = []
            for m in tf.getmembers():
                # Flexible match: Romanian side .ro file under ro-en pairing
                name = m.name
                if name.endswith(".ro") and ("ro-en" in name or "ro-en" in name.replace("/", "-")):
                    candidates.append(m)
                elif name.endswith("europarl-v7.ro-en.ro"):
                    candidates.append(m)

            if not candidates:
                names = ", ".join(x.name for x in tf.getmembers())
                raise FileNotFoundError(
                    f"Could not find Romanian side in Europarl archive. Searched for '*.ro' with 'ro-en'. Members: {names}"
                )
            # Prefer exact known name; otherwise pick the first candidate
            exact = [m for m in candidates if m.name.endswith("europarl-v7.ro-en.ro")]
            member_file = exact[0] if exact else candidates[0]
            f = tf.extractfile(member_file)
            if f is None:
                raise FileNotFoundError("Failed to extract Romanian side from archive")
            content = f.read().decode("utf-8", errors="ignore")

        norm = alphabet.normalize(content)
        with output_path.open("w", encoding="utf-8") as out:
            out.write(norm)
        return {
            "char_count": len(norm),
            "unique_chars": set(norm),
            "raw_size_bytes": raw_path.stat().st_size,
            "processed_size_bytes": output_path.stat().st_size,
        }


class OSCARExtractor(CorpusExtractor):
    def extract(self, raw_path: Path, output_path: Path, alphabet: Alphabet) -> dict[str, Any]:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            raise ImportError(
                "datasets is required for OSCARExtractor. Install 'datasets' and authenticate with HuggingFace."
            ) from e

        char_count = 0
        unique_chars: set[str] = set()
        with output_path.open("w", encoding="utf-8") as out:
            ds = load_dataset(
                "oscar-corpus/OSCAR-2201",
                language="ro",
                split="train",
                streaming=True,
                use_auth_token=True,
            )
            for record in ds:
                text = record.get("text", "")
                norm = alphabet.normalize(text)
                out.write(norm)
                char_count += len(norm)
                unique_chars.update(norm)

        return {
            "char_count": char_count,
            "unique_chars": unique_chars,
            "raw_size_bytes": raw_path.stat().st_size if raw_path.exists() else 0,
            "processed_size_bytes": output_path.stat().st_size,
        }


def get_extractor(extractor_name: str) -> CorpusExtractor:
    mapping: dict[str, type[CorpusExtractor]] = {
        "PlainTextExtractor": PlainTextExtractor,
        "WikipediaExtractor": WikipediaExtractor,
        "NLTKBrownExtractor": NLTKBrownExtractor,
        "NLTKGutenbergExtractor": NLTKGutenbergExtractor,
        "ZipTextExtractor": ZipTextExtractor,
        "GzipTextExtractor": GzipTextExtractor,
        "EuroparlExtractor": EuroparlExtractor,
        "OSCARExtractor": OSCARExtractor,
    }
    if extractor_name not in mapping:
        raise ValueError(f"Unknown extractor: {extractor_name}")
    return mapping[extractor_name]()



