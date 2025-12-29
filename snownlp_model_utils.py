import gzip
import importlib.util
import logging
import marshal
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class ReplaceResult:
    success: bool
    sentiment_dir: Optional[str]
    replaced_files: list[str]
    error: Optional[str]


@dataclass
class ModelFileStatus:
    name: str
    path: str
    exists: bool
    size: int
    mtime: Optional[float]
    mtime_str: Optional[str]
    backup_path: str
    backup_exists: bool
    valid: Optional[bool]
    validate_error: Optional[str]


def get_snownlp_package_dir() -> Optional[str]:
    spec = importlib.util.find_spec("snownlp")
    if spec is None:
        return None
    origin = getattr(spec, "origin", None)
    if not origin:
        return None
    return os.path.dirname(os.path.abspath(origin))


def get_snownlp_sentiment_dir() -> Optional[str]:
    pkg_dir = get_snownlp_package_dir()
    if not pkg_dir:
        return None
    return os.path.join(pkg_dir, "sentiment")


def _load_marshal_data(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read()

    try:
        raw = gzip.decompress(raw)
    except Exception:
        pass

    data = marshal.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("marshal data is not a dict")
    if "total" not in data or "d" not in data:
        raise ValueError("marshal dict missing required keys")
    if not isinstance(data.get("d"), dict):
        raise ValueError("marshal dict['d'] is not a dict")
    return data


def validate_marshal_model_file(path: str) -> Optional[str]:
    try:
        _load_marshal_data(path)
        return None
    except Exception as e:
        return str(e)


def restore_from_backup(sentiment_dir: str) -> ReplaceResult:
    replaced: list[str] = []
    try:
        for fname in ["sentiment.marshal.3", "sentiment.marshal"]:
            target = os.path.join(sentiment_dir, fname)
            backup = target + ".backup_gui"
            if os.path.exists(backup):
                shutil.copy2(backup, target)
                replaced.append(fname)

        if not replaced:
            return ReplaceResult(
                success=False,
                sentiment_dir=sentiment_dir,
                replaced_files=[],
                error="no .backup_gui files found",
            )

        return ReplaceResult(
            success=True,
            sentiment_dir=sentiment_dir,
            replaced_files=replaced,
            error=None,
        )
    except Exception as e:
        return ReplaceResult(
            success=False,
            sentiment_dir=sentiment_dir,
            replaced_files=replaced,
            error=str(e),
        )


def get_effective_model_filenames(replace_legacy_py2_file: bool = False) -> list[str]:
    targets: list[str] = []
    if sys.version_info[0] >= 3:
        targets.append("sentiment.marshal.3")
    else:
        targets.append("sentiment.marshal")

    if replace_legacy_py2_file and "sentiment.marshal" not in targets:
        targets.append("sentiment.marshal")
    return targets


def _format_mtime(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def collect_model_file_status(sentiment_dir: str, fname: str) -> ModelFileStatus:
    path = os.path.join(sentiment_dir, fname)
    exists = os.path.exists(path)
    size = 0
    mtime: Optional[float] = None
    if exists:
        try:
            size = int(os.path.getsize(path))
        except Exception:
            size = 0
        try:
            mtime = float(os.path.getmtime(path))
        except Exception:
            mtime = None

    backup_path = path + ".backup_gui"
    backup_exists = os.path.exists(backup_path)

    valid: Optional[bool] = None
    validate_error: Optional[str] = None
    if exists and (fname.endswith(".marshal") or fname.endswith(".marshal.3")):
        err = validate_marshal_model_file(path)
        if err:
            valid = False
            validate_error = err
        else:
            valid = True

    return ModelFileStatus(
        name=fname,
        path=path,
        exists=exists,
        size=size,
        mtime=mtime,
        mtime_str=_format_mtime(mtime),
        backup_path=backup_path,
        backup_exists=backup_exists,
        valid=valid,
        validate_error=validate_error,
    )


def collect_snownlp_model_status(replace_legacy_py2_file: bool = False) -> dict[str, Any]:
    spec = importlib.util.find_spec("snownlp")
    origin = getattr(spec, "origin", None) if spec else None
    pkg_dir = os.path.dirname(os.path.abspath(origin)) if origin else None
    sentiment_dir = os.path.join(pkg_dir, "sentiment") if pkg_dir else None

    targets = get_effective_model_filenames(replace_legacy_py2_file=replace_legacy_py2_file)
    files: list[ModelFileStatus] = []
    if sentiment_dir and os.path.isdir(sentiment_dir):
        for fname in targets:
            files.append(collect_model_file_status(sentiment_dir, fname))

    return {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "snownlp_origin": origin,
        "snownlp_pkg_dir": pkg_dir,
        "sentiment_dir": sentiment_dir,
        "targets": targets,
        "files": files,
    }


def log_model_status(logger: logging.Logger, status: dict[str, Any], event: str) -> None:
    try:
        logger.info(
            "snownlp_model_status event=%s python=%s snownlp_origin=%s sentiment_dir=%s targets=%s",
            event,
            status.get("python_executable"),
            status.get("snownlp_origin"),
            status.get("sentiment_dir"),
            status.get("targets"),
        )
        for f in status.get("files", []):
            if isinstance(f, ModelFileStatus):
                logger.info(
                    "snownlp_model_file event=%s name=%s exists=%s size=%s mtime=%s backup_exists=%s valid=%s err=%s",
                    event,
                    f.name,
                    f.exists,
                    f.size,
                    f.mtime_str,
                    f.backup_exists,
                    f.valid,
                    f.validate_error,
                )
    except Exception:
        return


def safe_replace_snownlp_model(
    source_model_file: str,
    sentiment_dir: Optional[str] = None,
    replace_legacy_py2_file: bool = False,
    logger: Optional[logging.Logger] = None,
) -> ReplaceResult:
    if sentiment_dir is None:
        sentiment_dir = get_snownlp_sentiment_dir()

    if not sentiment_dir or not os.path.isdir(sentiment_dir):
        return ReplaceResult(
            success=False,
            sentiment_dir=sentiment_dir,
            replaced_files=[],
            error="snownlp sentiment dir not found",
        )

    if not os.path.exists(source_model_file):
        return ReplaceResult(
            success=False,
            sentiment_dir=sentiment_dir,
            replaced_files=[],
            error=f"source model file not found: {source_model_file}",
        )

    err = validate_marshal_model_file(source_model_file)
    if err:
        return ReplaceResult(
            success=False,
            sentiment_dir=sentiment_dir,
            replaced_files=[],
            error=f"invalid source model file: {err}",
        )

    targets = get_effective_model_filenames(replace_legacy_py2_file=replace_legacy_py2_file)

    replaced: list[str] = []

    try:
        if logger:
            logger.info(
                "model_replace_begin src=%s sentiment_dir=%s targets=%s",
                os.path.abspath(source_model_file),
                os.path.abspath(sentiment_dir),
                targets,
            )
        for fname in targets:
            target_path = os.path.join(sentiment_dir, fname)
            backup_path = target_path + ".backup_gui"

            if os.path.exists(target_path) and not os.path.exists(backup_path):
                shutil.copy2(target_path, backup_path)
                if logger:
                    logger.info(
                        "model_backup_created target=%s backup=%s size=%s",
                        os.path.abspath(target_path),
                        os.path.abspath(backup_path),
                        os.path.getsize(backup_path) if os.path.exists(backup_path) else 0,
                    )

            tmp_path = target_path + ".tmp_new"
            shutil.copy2(source_model_file, tmp_path)

            tmp_err = validate_marshal_model_file(tmp_path)
            if tmp_err:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                raise ValueError(f"new model validation failed: {tmp_err}")

            os.replace(tmp_path, target_path)
            replaced.append(fname)
            if logger:
                try:
                    logger.info(
                        "model_replace_ok target=%s size=%s",
                        os.path.abspath(target_path),
                        os.path.getsize(target_path),
                    )
                except Exception:
                    pass

        return ReplaceResult(
            success=len(replaced) > 0,
            sentiment_dir=sentiment_dir,
            replaced_files=replaced,
            error=None,
        )
    except Exception as e:
        if logger:
            try:
                logger.exception(
                    "model_replace_exception src=%s sentiment_dir=%s targets=%s replaced=%s",
                    os.path.abspath(source_model_file),
                    os.path.abspath(sentiment_dir) if sentiment_dir else None,
                    targets,
                    replaced,
                )
            except Exception:
                pass
        try:
            restore_from_backup(sentiment_dir)
            if logger:
                logger.info("model_replace_rollback_ok sentiment_dir=%s", os.path.abspath(sentiment_dir))
        except Exception:
            pass
        return ReplaceResult(
            success=False,
            sentiment_dir=sentiment_dir,
            replaced_files=replaced,
            error=str(e),
        )
