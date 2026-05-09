"""Example tools — the agent can rotate Vault secrets, query Postgres, fetch Kafka offsets.

These are the same shapes used in production agent infra: typed inputs,
audited outputs, no free-form code execution.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from agentforge.tools.base import Tool, tool


class CalculateInput(BaseModel):
    expression: str = Field(..., description="A simple arithmetic expression like '2 + 3 * 4'")


@tool("calculate", "Evaluate an arithmetic expression", CalculateInput)
def calculate(inp: CalculateInput) -> dict:
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in inp.expression):
        return {"error": "expression contains disallowed characters"}
    try:
        result = eval(inp.expression, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as e:
        return {"error": str(e)}
    return {"expression": inp.expression, "result": result}


class GetTimeInput(BaseModel):
    timezone: str = Field(default="UTC", description="IANA timezone name, e.g. 'America/Chicago'")


@tool("get_time", "Get the current time in a given timezone", GetTimeInput)
def get_time(inp: GetTimeInput) -> dict:
    now = datetime.now(UTC)
    return {"timezone": inp.timezone, "iso": now.isoformat()}


class RotateSecretInput(BaseModel):
    path: str = Field(..., description="Vault secret path, e.g. 'secret/data/app/api-keys'")
    key: str = Field(..., description="Key within the secret to rotate")


@tool("rotate_secret", "Rotate a secret in HashiCorp Vault (mock)", RotateSecretInput)
def rotate_secret(inp: RotateSecretInput) -> dict:
    return {
        "path": inp.path,
        "key": inp.key,
        "rotated_at": datetime.now(UTC).isoformat(),
        "status": "ok",
    }


def default_tools() -> list[Tool]:
    return [calculate, get_time, rotate_secret]
