"""
Co-STEER 表达式转译器

将 LLM 生成的因子表达式字符串转译为 Node 表达式树，
复用 gp_factor_miner.py 的 Node 数据结构。
"""

import re
from typing import Optional, List, Tuple

from core.alpha.gp_factor_miner import (
    Node,
    UNARY_TS,
    BINARY_TS,
    CROSS_SECTIONAL,
    UNARY_ARITHMETIC,
    BINARY_ARITHMETIC,
)
from core.alpha.hypothesis_generator import VALID_TERMINALS, VALID_WINDOWS


class ParseError(Exception):
    pass


def tokenize(expr: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c in "(),":
            tokens.append(c)
            i += 1
            continue
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", expr[i:])
        if match:
            tokens.append(match.group())
            i += len(match.group())
            continue
        match = re.match(r"\d+", expr[i:])
        if match:
            tokens.append(match.group())
            i += len(match.group())
            continue
        raise ParseError(f"Unexpected character at position {i}: {c!r} in expr: {expr}")
    return tokens


class _Parser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, expected: str):
        tok = self.consume()
        if tok != expected:
            raise ParseError(f"Expected {expected!r}, got {tok!r} at pos {self.pos - 1}")

    def parse_expr(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected end of input")

        if tok == "(":
            raise ParseError(f"Unexpected ( at pos {self.pos}")

        name = self.consume()

        if self.peek() == "(":
            return self._parse_call(name)
        else:
            return self._parse_terminal(name)

    def _parse_terminal(self, name: str) -> Node:
        if name not in VALID_TERMINALS:
            raise ParseError(
                f"Unknown terminal: {name!r}. Valid: {VALID_TERMINALS}"
            )
        return Node(op="terminal", value=name)

    def _parse_call(self, func_name: str) -> Node:
        self.expect("(")

        if func_name in UNARY_TS:
            child = self.parse_expr()
            self.expect(",")
            window = int(self.consume())
            if window not in VALID_WINDOWS:
                raise ParseError(
                    f"Invalid window {window} for {func_name}. Valid: {VALID_WINDOWS}"
                )
            self.expect(")")
            return Node(op=func_name, children=[child], window=window)

        if func_name in BINARY_TS:
            child1 = self.parse_expr()
            self.expect(",")
            child2 = self.parse_expr()
            self.expect(",")
            window = int(self.consume())
            if window not in VALID_WINDOWS:
                raise ParseError(
                    f"Invalid window {window} for {func_name}. Valid: {VALID_WINDOWS}"
                )
            self.expect(")")
            return Node(op=func_name, children=[child1, child2], window=window)

        if func_name in CROSS_SECTIONAL:
            child = self.parse_expr()
            self.expect(")")
            return Node(op=func_name, children=[child])

        if func_name in UNARY_ARITHMETIC:
            child = self.parse_expr()
            self.expect(")")
            return Node(op=func_name, children=[child])

        if func_name in BINARY_ARITHMETIC:
            child1 = self.parse_expr()
            self.expect(",")
            child2 = self.parse_expr()
            self.expect(")")
            return Node(op=func_name, children=[child1, child2])

        raise ParseError(
            f"Unknown operator: {func_name!r}. "
            f"Valid: {UNARY_TS + BINARY_TS + CROSS_SECTIONAL + UNARY_ARITHMETIC + BINARY_ARITHMETIC}"
        )


def parse_expression(expr_str: str) -> Node:
    expr_str = expr_str.strip()
    tokens = tokenize(expr_str)
    parser = _Parser(tokens)
    node = parser.parse_expr()
    if parser.pos != len(tokens):
        raise ParseError(
            f"Trailing tokens at pos {parser.pos}: {tokens[parser.pos:]}"
        )
    return node


def translate_hypothesis(hypothesis: dict) -> Tuple[Optional[Node], str]:
    expr_str = hypothesis.get("expr", "").strip()
    if not expr_str:
        expr_str = hypothesis.get("formula", "").strip()
    if not expr_str:
        return None, "empty expression"

    try:
        node = parse_expression(expr_str)
        return node, ""
    except ParseError as e:
        return None, str(e)


def translate_batch(hypotheses: List[dict]) -> List[Tuple[dict, Node]]:
    results = []
    for hyp in hypotheses:
        node, err = translate_hypothesis(hyp)
        if node is not None:
            results.append((hyp, node))
        else:
            print(f"[Translator] Skip {hyp.get('factor_id', '?')}: {err}")
    return results
