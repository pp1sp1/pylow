"""Static type inference analyzer for Python AST.

Scans function bodies to infer variable types from assignments,
enabling the compiler to use native LLVM types instead of boxed values
for monomorphic variables.
"""

from __future__ import annotations

import ast
from typing import Dict, Set

from .types import PyType


class StaticTypeAnalyzer(ast.NodeVisitor):
    """Infers static types for local variables by scanning AST assignments.
    
    Collects all potential type assignments for each variable. If a variable
    receives exactly one type across all assignments (monomorphic), the compiler
    can use a native LLVM type instead of boxing.
    
    Attributes:
        var_types: Mapping from variable name to set of inferred PyTypes.
    """

    def __init__(self) -> None:
        self.var_types: Dict[str, Set[PyType]] = {}

    def infer_expr(self, node: ast.AST) -> PyType:
        """Infer the PyType of an AST expression node.
        
        Args:
            node: The AST expression to analyze.
            
        Returns:
            The inferred PyType, or PyType.OBJECT for unknown/complex expressions.
        """
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return PyType.INT
            if isinstance(node.value, float):
                return PyType.FLOAT
            if isinstance(node.value, str):
                return PyType.STR
            if isinstance(node.value, bool):
                return PyType.BOOL

        elif isinstance(node, ast.List):
            return PyType.LIST
        elif isinstance(node, ast.Set):
            return PyType.SET
        elif isinstance(node, ast.Tuple):
            return PyType.TUPLE
        elif isinstance(node, ast.JoinedStr):
            return PyType.STR
        elif isinstance(node, ast.Dict):
            return PyType.DICT

        # Improved inference for binary operations
        elif isinstance(node, ast.BinOp):
            left_t = self.infer_expr(node.left)
            right_t = self.infer_expr(node.right)
            # Division (/) always produces FLOAT in Python
            if isinstance(node.op, ast.Div):
                return PyType.FLOAT
            # If both operands are INT, the result is INT
            if left_t == PyType.INT and right_t == PyType.INT:
                return PyType.INT
            # If either operand is FLOAT, the result is FLOAT
            if left_t == PyType.FLOAT or right_t == PyType.FLOAT:
                return PyType.FLOAT
            return PyType.OBJECT

        # Handle Name references — return known type if monomorphic
        elif isinstance(node, ast.Name) and node.id in self.var_types:
            if len(self.var_types[node.id]) == 1:
                return list(self.var_types[node.id])[0]

        return PyType.OBJECT

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit an assignment node and record type information for targets.
        
        Args:
            node: The ast.Assign node to analyze.
        """
        t = self.infer_expr(node.value)
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                if tgt.id not in self.var_types:
                    self.var_types[tgt.id] = set()
                self.var_types[tgt.id].add(t)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Visit an augmented assignment node (e.g., x += 1).
        
        For arithmetic operations on INT variables, preserves the INT type.
        For other cases, falls back to OBJECT to allow type changes.
        
        Args:
            node: The ast.AugAssign node to analyze.
        """
        if isinstance(node.target, ast.Name):
            if node.target.id not in self.var_types:
                self.var_types[node.target.id] = set()

            # Attempt to infer type from the operation
            expr_type = self.infer_expr(node.value)

            # Arithmetic operators on int preserve INT type
            int_ops = (
                ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod,
                ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
            )

            if expr_type == PyType.INT and isinstance(node.op, int_ops):
                # If target already has INT in its type set, preserve it
                if PyType.INT in self.var_types[node.target.id]:
                    pass  # Don't add OBJECT — keep it monomorphic
                else:
                    self.var_types[node.target.id].add(expr_type)
            else:
                # Fallback for other cases (e.g., i += "abc")
                self.var_types[node.target.id].add(PyType.OBJECT)

        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Visit a for-loop node and record type info for loop variables.
        
        Variables iterating over range() are inferred as INT;
        others default to OBJECT.
        
        Args:
            node: The ast.For node to analyze.
        """
        if isinstance(node.target, ast.Name):
            if node.target.id not in self.var_types:
                self.var_types[node.target.id] = set()
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ):
                self.var_types[node.target.id].add(PyType.INT)
            else:
                self.var_types[node.target.id].add(PyType.OBJECT)
        self.generic_visit(node)
