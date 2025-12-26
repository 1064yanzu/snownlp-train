# -*- coding: utf-8 -*-

import csv
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class ReadCsvResult:
    df: pd.DataFrame
    encoding: str
    sep: str


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "content": "content",
        "text": "content",
        "sentence": "content",
        "review": "content",
        "comment": "content",
        "文本": "content",
        "内容": "content",
        "评论": "content",
        "句子": "content",
        "sentiment": "sentiment",
        "label": "sentiment",
        "class": "sentiment",
        "情感": "sentiment",
        "标签": "sentiment",
    }

    renamed = {}
    for c in df.columns:
        key = str(c).strip()
        renamed[c] = mapping.get(key, mapping.get(key.lower(), key))

    return df.rename(columns=renamed)


def _sniff_sep(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def read_sentiment_csv(
    path: str,
    encodings: Optional[Iterable[str]] = None,
    sep: Optional[str] = None,
) -> ReadCsvResult:
    encodings = list(encodings or ["utf-8-sig", "utf-8", "gbk", "gb2312", "gb18030"])

    with open(path, "rb") as f:
        raw = f.read()

    if not raw:
        raise ValueError(f"CSV 文件为空: {path}")

    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            text = raw.decode(enc)
            sample = "\n".join(text.splitlines()[:50])
            sniffed_sep = sep or _sniff_sep(sample)

            df = pd.read_csv(
                path,
                encoding=enc,
                sep=sniffed_sep,
                engine="python",
            )
            df = _normalize_columns(df)

            if "content" not in df.columns or "sentiment" not in df.columns:
                raise ValueError(f"缺少必要列 content/sentiment，实际列: {list(df.columns)}")

            return ReadCsvResult(df=df, encoding=enc, sep=sniffed_sep)
        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"无法读取CSV文件（已尝试编码: {encodings}）: {path}. 最后错误: {last_err}")
