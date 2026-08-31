"""Multi-turn tool-use environment framework.

An Episode is a chat: system(task instructions) -> user(instance) ->
assistant(action) -> user(observation) -> ... until submit / max_turns.

Actions are fenced JSON blocks:
    ```action
    {"tool": "sql", "args": {"query": "SELECT ..."}}
    ```
The last well-formed action block in the assistant message is executed.
A malformed / missing action block yields a format-error observation
(which counts as an err_event for tracing).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

ACTION_RE = re.compile(r"```action\s*\n(.*?)```", re.DOTALL)

FORMAT_ERROR_OBS = (
    "FormatError: no valid ```action``` block with JSON "
    '{"tool": ..., "args": {...}} found in your reply. '
    "Reply with exactly one action block."
)


@dataclass
class Turn:
    role: str            # "assistant" | "user"
    content: str
    is_error_obs: bool = False   # observation contains a tool/format error


@dataclass
class Episode:
    task: str
    instance_id: str
    system: str
    user0: str
    turns: list[Turn] = field(default_factory=list)
    done: bool = False
    success: bool = False
    submitted: Any = None
    n_actions: int = 0
    meta: dict = field(default_factory=dict)

    def messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": self.user0}]
        for t in self.turns:
            msgs.append({"role": t.role, "content": t.content})
        return msgs


def parse_action(text: str):
    """Return (tool, args) from the LAST action block, or None."""
    blocks = ACTION_RE.findall(text)
    if not blocks:
        return None
    try:
        obj = json.loads(blocks[-1].strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None
    args = obj.get("args", {}) or {}
    if not isinstance(args, dict):
        # models sometimes emit args as a bare value; treat it as the answer
        args = {"answer": args}
    return obj["tool"], args


class TaskInstance:
    """One concrete instance of a task family.

    Subclasses implement tools() and verify(). Tool callables take **args
    and return (observation:str, is_error:bool).
    """
    task: str = "base"
    max_turns: int = 8

    def __init__(self, instance_id: str, seed: int, tier: int = 0):
        self.instance_id = instance_id
        self.seed = seed
        self.tier = tier

    def system_prompt(self) -> str:
        raise NotImplementedError

    def user_prompt(self) -> str:
        raise NotImplementedError

    def tools(self) -> dict[str, Callable]:
        raise NotImplementedError

    def verify(self, submission: Any) -> bool:
        raise NotImplementedError


def env_step(inst: TaskInstance, ep: Episode, assistant_text: str) -> None:
    """Apply one assistant message to the episode: parse, run tool, append
    observation. Sets ep.done on submit / turn budget exhaustion."""
    ep.turns.append(Turn("assistant", assistant_text))
    ep.n_actions += 1
    parsed = parse_action(assistant_text)
    if parsed is None:
        obs, is_err = FORMAT_ERROR_OBS, True
    else:
        tool, args = parsed
        if tool == "submit":
            ep.submitted = args.get("answer")
            ep.done = True
            try:
                ep.success = bool(inst.verify(ep.submitted))
            except Exception:
                ep.success = False
            return
        fn = inst.tools().get(tool)
        if fn is None:
            obs, is_err = f"ToolError: unknown tool '{tool}'.", True
        else:
            try:
                obs, is_err = fn(**args)
            except TypeError as e:
                obs, is_err = f"ToolError: bad arguments: {e}", True
            except Exception as e:
                obs, is_err = f"ToolError: {type(e).__name__}: {e}", True
    if len(obs) > 2000:
        obs = obs[:2000] + "\n...[truncated]"
    ep.turns.append(Turn("user", obs, is_error_obs=is_err))
    if ep.n_actions >= inst.max_turns:
        ep.done = True
        ep.success = False
