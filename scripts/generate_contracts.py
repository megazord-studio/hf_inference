"""Generate backend and frontend contract models from proto schema.

Usage:
    uv run python scripts/generate_contracts.py

This script keeps `app/schemas_pb2.py` and `frontend/generated/contracts_pb.ts`
in sync with `proto/contracts.proto`, ensuring the protobuf definition is the
single source of truth for shared contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_PATH = REPO_ROOT / "proto" / "contracts.proto"
PY_OUT = REPO_ROOT / "app" / "schemas_pb2.py"
TS_OUT = REPO_ROOT / "frontend" / "generated" / "contracts_pb.ts"

FIELD_PATTERN = re.compile(r"^(repeated\s+)?([\w\.]+)\s+(\w+)\s*=\s*\d+;")

PY_TYPE_MAP: Dict[str, str] = {
    "string": "str",
    "int32": "int",
    "int64": "int",
    "double": "float",
    "bool": "bool",
    "google.protobuf.Struct": "Dict[str, Any]",
    "google.protobuf.Value": "Any",
}
TS_TYPE_MAP: Dict[str, str] = {
    "string": "string",
    "int32": "number",
    "int64": "number",
    "double": "number",
    "bool": "boolean",
    "google.protobuf.Struct": "Record<string, unknown>",
    "google.protobuf.Value": "unknown",
}

REQUIRED_FIELDS = {
    ("ErrorResponse", "code"),
    ("ErrorResponse", "message"),
    ("TaskOutputMetadata", "task"),
    ("InferenceResult", "task_output"),
    ("InferenceResult", "echo"),
    ("InferenceResult", "info"),
    ("InferenceResult", "metadata"),
    ("ModelMeta", "id"),
    ("InferenceErrorPayload", "error"),
}

MESSAGE_NAMES: Set[str] = set()

@dataclass
class FieldDef:
    name: str
    type_name: str
    repeated: bool = False

@dataclass
class MessageDef:
    name: str
    fields: List[FieldDef]


def parse_proto() -> List[MessageDef]:
    messages: List[MessageDef] = []
    current: MessageDef | None = None
    for raw_line in PROTO_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("message "):
            name = line.split()[1]
            current = MessageDef(name=name, fields=[])
            MESSAGE_NAMES.add(name)
            continue
        if line == "}" and current:
            messages.append(current)
            current = None
            continue
        if current is None:
            continue
        match = FIELD_PATTERN.match(line)
        if not match:
            continue
        repeated, type_name, field_name = match.groups()
        current.fields.append(
            FieldDef(name=field_name, type_name=type_name, repeated=bool(repeated))
        )
    return messages


def _is_required(message: str, field: FieldDef) -> bool:
    if field.repeated:
        return True
    return (message, field.name) in REQUIRED_FIELDS


def _py_type(message: str, field: FieldDef) -> str:
    base = PY_TYPE_MAP.get(field.type_name)
    if base is None:
        base = field.type_name if field.type_name in MESSAGE_NAMES else "Any"
    annotated = f"List[{base}]" if field.repeated else base
    if field.repeated or _is_required(message, field):
        return annotated
    return f"Optional[{annotated}]"


def _ts_type(message: str, field: FieldDef) -> str:
    base = TS_TYPE_MAP.get(field.type_name)
    if base is None:
        base = field.type_name if field.type_name in MESSAGE_NAMES else "unknown"
    annotated = f"{base}[]" if field.repeated else base
    return annotated


def generate_python(messages: List[MessageDef]) -> None:
    needed_types = {"BaseModel"}
    for msg in messages:
        for field in msg.fields:
            py = PY_TYPE_MAP.get(field.type_name)
            if py in {"Dict[str, Any]", "Any"}:
                needed_types.update({"Dict", "Any"})
            if field.repeated:
                needed_types.add("List")
            if not field.repeated:
                needed_types.add("Optional")
    imports = ["Any", "Dict", "List", "Optional"]
    imports = [name for name in imports if name in needed_types]

    header = textwrap.dedent(
        """Auto-generated from proto/contracts.proto.

        Do not edit manually; run `poe generate-contracts` instead."""
    ).strip()

    needs_field = any(field.repeated for msg in messages for field in msg.fields)
    pydantic_import = "from pydantic import BaseModel, Field" if needs_field else "from pydantic import BaseModel"

    lines = [
        '"""Auto-generated from proto/contracts.proto.\n\nDo not edit manually; run `poe generate-contracts` instead."""',
        "from __future__ import annotations",
        "",
    ]
    if imports:
        lines.append(f"from typing import {', '.join(imports)}")
        lines.append("")
    lines.append(pydantic_import)
    lines.append("")

    for message in messages:
        lines.append(f"class {message.name}(BaseModel):")
        if not message.fields:
            lines.append("    pass")
            lines.append("")
            continue
        for field in message.fields:
            annotation = _py_type(message.name, field)
            if field.repeated:
                lines.append(f"    {field.name}: {annotation} = Field(default_factory=list)")
            elif _is_required(message.name, field):
                lines.append(f"    {field.name}: {annotation}")
            else:
                lines.append(f"    {field.name}: {annotation} = None")
        lines.append("")

    exports = ", ".join(f'"{m.name}"' for m in messages)
    lines.append(f"__all__ = [{exports}]")
    lines.append("")
    PY_OUT.write_text("\n".join(lines))


def generate_ts(messages: List[MessageDef]) -> None:
    header = "// Auto-generated from proto/contracts.proto. Do not edit manually."
    lines = [header, ""]
    for message in messages:
        lines.append(f"export interface {message.name} {{")
        if not message.fields:
            lines.append("}")
            lines.append("")
            continue
        for field in message.fields:
            ts_type = _ts_type(message.name, field)
            optional = "" if _is_required(message.name, field) else "?"
            nullable = " | null" if not field.repeated else ""
            lines.append(f"  {field.name}{optional}: {ts_type}{nullable};")
        lines.append("}")
        lines.append("")
    TS_OUT.write_text("\n".join(lines))


def main() -> None:
    if not PROTO_PATH.exists():
        raise SystemExit(f"Proto file not found at {PROTO_PATH}")
    messages = parse_proto()
    if not messages:
        raise SystemExit("No messages found in proto schema")
    generate_python(messages)
    generate_ts(messages)
    print(
        "Generated app/schemas_pb2.py and frontend/generated/contracts_pb.ts "
        "from proto/contracts.proto"
    )


if __name__ == "__main__":
    main()
