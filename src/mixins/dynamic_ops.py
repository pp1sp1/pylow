"""Dynamic (runtime) arithmetic and comparison operations for boxed values."""

from __future__ import annotations

import ast
import sys
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING

import llvmlite.ir as ir
import llvmlite.binding as llvm

from ..types import (
    Tag, PyType, EXCEPTION_TYPES,
    I8, I32, I64, F64, I1, I8P, VOID,
    GC_HEADER_TY, BOXED_TY, BOXED_PTR,
    LIST_TY, LIST_PTR, ENTRY_TY, ENTRY_PTR,
    DICT_TY, DICT_PTR, STR_TY, STR_PTR,
    CLASS_TY, CLASS_PTR, INSTANCE_TY, INSTANCE_PTR,
    ITER_DATA_TY, ITER_DATA_PTR,
    SZ_GC_HEADER, SZ_BOXED, SZ_ENTRY, SZ_LIST, SZ_DICT, SZ_STR, SZ_ITER,
    pytype_to_llvm,
)
from ..exceptions import CompileError
from ..symbols import VarInfo, SymbolTable
from ..values import Value
from ..type_analyzer import StaticTypeAnalyzer

if TYPE_CHECKING:
    pass


class DynamicOpsMixin:
    """Dynamic (runtime) arithmetic and comparison operations for boxed values."""

    def _pay_to_f64_runtime(self, tag: ir.Value, pay: ir.Value) -> ir.Value:
        """Konwertuje (tag, payload) → double w runtime."""
        res = self.builder.alloca(F64, name="p2f")
        is_f = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.FLOAT))
        f_bb = self.current_func.append_basic_block("p2f.fp")
        i_bb = self.current_func.append_basic_block("p2f.int")
        c_bb = self.current_func.append_basic_block("p2f.cont")
        self.builder.cbranch(is_f, f_bb, i_bb)

        self.builder.position_at_end(f_bb)
        self.builder.store(self.builder.bitcast(pay, F64), res)
        self.builder.branch(c_bb)

        self.builder.position_at_end(i_bb)
        self.builder.store(self.builder.sitofp(pay, F64), res)
        self.builder.branch(c_bb)

        self.builder.position_at_end(c_bb)
        return self.builder.load(res)

    def dynamic_binop(
        self,
        op: ast.operator,
        ltag: ir.Value,
        lpay: ir.Value,
        rtag: ir.Value,
        rpay: ir.Value,
    ) -> Value:
        """
        Arytmetyka na parze (tag, payload) z obsługą dynamicznego typowania.
        Obsługuje: INT + INT, STR + STR, LIST + LIST oraz fallback do FLOAT.
        Zwraca Value(BOXED_PTR, PyType.OBJECT).
        """
        res_pay = self.builder.alloca(I64, name="dbo_pay")
        res_tag = self.builder.alloca(I64, name="dbo_tag")

        self.builder.store(ir.Constant(I64, 0), res_pay)
        self.builder.store(ir.Constant(I64, Tag.INT), res_tag)

        both_int = self.builder.and_(
            self.builder.icmp_signed("==", ltag, ir.Constant(I64, Tag.INT)),
            self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.INT)),
        )
        both_str = self.builder.and_(
            self.builder.icmp_signed("==", ltag, ir.Constant(I64, Tag.STR)),
            self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.STR)),
        )
        both_lst = self.builder.or_(
            self.builder.and_(
                self.builder.icmp_signed("==", ltag, ir.Constant(I64, Tag.LIST)),
                self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.LIST)),
            ),
            self.builder.or_(
                self.builder.and_(
                    self.builder.icmp_signed("==", ltag, ir.Constant(I64, Tag.SET)),
                    self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.SET)),
                ),
                self.builder.or_(
                    self.builder.and_(
                        self.builder.icmp_signed("==", ltag, ir.Constant(I64, Tag.LIST)),
                        self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.SET)),
                    ),
                    self.builder.and_(
                        self.builder.icmp_signed("==", ltag, ir.Constant(I64, Tag.SET)),
                        self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.LIST)),
                    ),
                ),
            ),
        )
        # Check if either operand is a SET (for result type)
        either_set = self.builder.or_(
            self.builder.icmp_signed("==", ltag, ir.Constant(I64, Tag.SET)),
            self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.SET)),
        )

        int_bb = self.current_func.append_basic_block("dbo.int")
        not_int_bb = self.current_func.append_basic_block("dbo.not_int")
        str_bb = self.current_func.append_basic_block("dbo.str")
        not_str_bb = self.current_func.append_basic_block("dbo.not_str")
        lst_bb = self.current_func.append_basic_block("dbo.lst")
        flt_bb = self.current_func.append_basic_block("dbo.flt")
        end_bb = self.current_func.append_basic_block("dbo.end")

        if isinstance(op, ast.Div):
            self.builder.branch(flt_bb)
        else:
            self.builder.cbranch(both_int, int_bb, not_int_bb)

        self.builder.position_at_end(int_bb)
        ir_val = self._static_int_binop(op, lpay, rpay)
        self.builder.store(ir_val, res_pay)
        self.builder.store(ir.Constant(I64, Tag.INT), res_tag)
        self.builder.branch(end_bb)

        self.builder.position_at_end(not_int_bb)
        self.builder.cbranch(both_str, str_bb, not_str_bb)

        self.builder.position_at_end(str_bb)
        s1 = Value(self.builder.inttoptr(lpay, STR_PTR), PyType.STR)
        s2 = Value(self.builder.inttoptr(rpay, STR_PTR), PyType.STR)
        concat_val = self.concat_strings(s1, s2)
        self.builder.store(self.builder.ptrtoint(concat_val.llvm, I64), res_pay)
        self.builder.store(ir.Constant(I64, Tag.STR), res_tag)
        self.builder.branch(end_bb)

        self.builder.position_at_end(not_str_bb)
        self.builder.cbranch(both_lst, lst_bb, flt_bb)

        self.builder.position_at_end(lst_bb)
        l_val = Value(self.builder.inttoptr(lpay, LIST_PTR), PyType.LIST)
        r_val = Value(self.builder.inttoptr(rpay, LIST_PTR), PyType.LIST)
        # FIX: Obsługa operacji na zbiorach dla typów OBJECT (boxed)
        is_set_op = isinstance(op, (ast.BitOr, ast.BitAnd, ast.Sub, ast.BitXor))
        if isinstance(op, ast.BitOr):
            res_lst = self._set_union(l_val, r_val)
        elif isinstance(op, ast.BitAnd):
            res_lst = self._set_intersection(l_val, r_val)
        elif isinstance(op, ast.Sub):
            res_lst = self._set_difference(l_val, r_val)
        elif isinstance(op, ast.BitXor):
            res_lst = self._set_symmetric_difference(l_val, r_val)
        elif isinstance(op, ast.Add):
            res_lst = self._concat_lists(l_val, r_val)
        else:
            # Fallback: concatenation
            res_lst = self._concat_lists(l_val, r_val)
        self.builder.store(self.builder.ptrtoint(res_lst.llvm, I64), res_pay)
        # FIX: Jeśli przynajmniej jeden operand to SET i operacja to |,&,-,^, wynik to SET
        result_tag_lst_bb = self.current_func.append_basic_block("dbo.lst_tag")
        result_tag_set_bb = self.current_func.append_basic_block("dbo.set_tag")
        result_tag_done_bb = self.current_func.append_basic_block("dbo.tag_done")
        if is_set_op:
            self.builder.cbranch(either_set, result_tag_set_bb, result_tag_lst_bb)
        else:
            self.builder.branch(result_tag_lst_bb)
        self.builder.position_at_end(result_tag_lst_bb)
        self.builder.store(ir.Constant(I64, Tag.LIST), res_tag)
        self.builder.branch(result_tag_done_bb)
        self.builder.position_at_end(result_tag_set_bb)
        self.builder.store(ir.Constant(I64, Tag.SET), res_tag)
        self.builder.branch(result_tag_done_bb)
        self.builder.position_at_end(result_tag_done_bb)
        self.builder.branch(end_bb)

        self.builder.position_at_end(flt_bb)
        lf = self._pay_to_f64_runtime(ltag, lpay)
        rf = self._pay_to_f64_runtime(rtag, rpay)
        ff = self._static_float_binop(op, lf, rf)
        self.builder.store(self.builder.bitcast(ff, I64), res_pay)
        self.builder.store(ir.Constant(I64, Tag.FLOAT), res_tag)
        self.builder.branch(end_bb)

        self.builder.position_at_end(end_bb)
        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)])
        bv = self.builder.bitcast(raw, BOXED_PTR)
        z = ir.Constant(I32, 0)
        null_i8p = ir.Constant(I8P, None)
        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 0)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 1)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 2)], inbounds=True),
        )
        self.builder.store(
            null_i8p, self.builder.gep(bv, [z, z, ir.Constant(I32, 3)], inbounds=True)
        )
        self.builder.store(
            self.builder.load(res_tag),
            self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True),
        )
        self.builder.store(
            self.builder.load(res_pay),
            self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True),
        )
        return Value(bv, PyType.OBJECT)

    def _static_int_binop(self, op: ast.operator, l: ir.Value, r: ir.Value) -> ir.Value:
        if isinstance(op, ast.Add):
            return self.builder.add(l, r)
        if isinstance(op, ast.Sub):
            return self.builder.sub(l, r)
        if isinstance(op, ast.Mult):
            return self.builder.mul(l, r)
        if isinstance(op, ast.FloorDiv):
            return self.builder.sdiv(l, r)
        if isinstance(op, ast.Mod):
            return self.builder.srem(l, r)
        if isinstance(op, ast.Div):
            return self.builder.sdiv(l, r)
        if isinstance(op, ast.BitOr):
            return self.builder.or_(l, r)
        if isinstance(op, ast.BitAnd):
            return self.builder.and_(l, r)
        if isinstance(op, ast.BitXor):
            return self.builder.xor(l, r)
        if isinstance(op, ast.LShift):
            return self.builder.shl(l, r)
        if isinstance(op, ast.RShift):
            return self.builder.ashr(l, r)
        if isinstance(op, ast.Pow):
            # Simple integer power: use repeated multiplication
            # Only works for small non-negative exponents
            base = l
            exp = r
            if isinstance(exp, ir.Constant):
                exp_val = exp.constant
                if exp_val == 0:
                    return ir.Constant(I64, 1)
                elif exp_val == 1:
                    return base
                else:
                    result = base
                    for _ in range(int(exp_val) - 1):
                        result = self.builder.mul(result, base)
                    return result
            # Fallback: call pow runtime function or use loop
            return self.builder.mul(l, r)
        raise CompileError(f"dynamic int binop: {type(op).__name__}")

    def _static_float_binop(
        self, op: ast.operator, l: ir.Value, r: ir.Value
    ) -> ir.Value:
        if isinstance(op, ast.Add):
            return self.builder.fadd(l, r)
        if isinstance(op, ast.Sub):
            return self.builder.fsub(l, r)
        if isinstance(op, ast.Mult):
            return self.builder.fmul(l, r)
        if isinstance(op, ast.Div):
            return self.builder.fdiv(l, r)
        if isinstance(op, ast.Mod):
            return self.builder.frem(l, r)
        if isinstance(op, ast.FloorDiv):
            return self.builder.sitofp(
                self.builder.fptosi(self.builder.fdiv(l, r), I64), F64
            )
        if isinstance(op, ast.Pow):
            pow_fn = self.functions.get("llvm.pow.f64")
            if pow_fn is None:
                fty = ir.FunctionType(F64, [F64, F64])
                pow_fn = ir.Function(self.module, fty, name="llvm.pow.f64")
                self.functions["llvm.pow.f64"] = pow_fn
            return self.builder.call(pow_fn, [l, r])
        # Bitwise ops na float – konwertuj do int, wykonaj, wróć do float
        if isinstance(op, (ast.BitOr, ast.BitAnd, ast.BitXor, ast.LShift, ast.RShift)):
            li = self.builder.fptosi(l, I64)
            ri = self.builder.fptosi(r, I64)
            if isinstance(op, ast.BitOr):
                res = self.builder.or_(li, ri)
            elif isinstance(op, ast.BitAnd):
                res = self.builder.and_(li, ri)
            elif isinstance(op, ast.BitXor):
                res = self.builder.xor(li, ri)
            elif isinstance(op, ast.LShift):
                res = self.builder.shl(li, ri)
            elif isinstance(op, ast.RShift):
                res = self.builder.ashr(li, ri)
            return self.builder.sitofp(res, F64)
        raise CompileError(f"dynamic float binop: {type(op).__name__}")

    # ──────────────────────────────────────────────────────────────
    #  DYNAMIC COMPARE
    # ──────────────────────────────────────────────────────────────

    def dynamic_compare(
        self,
        op: ast.cmpop,
        ltag: ir.Value,
        lpay: ir.Value,
        rtag: ir.Value,
        rpay: ir.Value,
    ) -> Value:
        """Porównanie w runtime. Zwraca BOOL (i1)."""
        if type(op) in (ast.In, ast.NotIn):
            # Use conditional branches to avoid calling wrong runtime functions
            # (calling list_contains with a string pointer causes UB/infinite loop)
            is_lst = self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.LIST))
            is_tpl = self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.TUPLE))
            is_dct = self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.DICT))
            is_str = self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.STR))
            # TUPLE and SET have same structure as LIST, so treat them the same
            is_lst_or_tpl = self.builder.or_(is_lst, is_tpl)
            is_set = self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.SET))
            is_lst_like = self.builder.or_(is_lst_or_tpl, is_set)

            lst_bb = self.current_func.append_basic_block("in.lst")
            not_lst_bb = self.current_func.append_basic_block("in.not_lst")
            dct_bb = self.current_func.append_basic_block("in.dct")
            not_dct_bb = self.current_func.append_basic_block("in.not_dct")
            str_bb = self.current_func.append_basic_block("in.str")
            other_bb = self.current_func.append_basic_block("in.other")
            merge_bb = self.current_func.append_basic_block("in.merge")

            # Chain: is_lst_like? → lst : (is_dct? → dct : (is_str? → str : other))
            self.builder.cbranch(is_lst_like, lst_bb, not_lst_bb)

            self.builder.position_at_end(not_lst_bb)
            self.builder.cbranch(is_dct, dct_bb, not_dct_bb)

            self.builder.position_at_end(not_dct_bb)
            self.builder.cbranch(is_str, str_bb, other_bb)

            # LIST branch
            self.builder.position_at_end(lst_bb)
            list_cont_fn = self._ensure_runtime_func(
                "__py2llvm_list_contains", ir.FunctionType(I1, [LIST_PTR, I64, I64])
            )
            res_l = self.builder.call(
                list_cont_fn, [self.builder.inttoptr(rpay, LIST_PTR), ltag, lpay]
            )
            self.builder.branch(merge_bb)

            # DICT branch
            self.builder.position_at_end(dct_bb)
            dict_cont_fn = self._ensure_runtime_func(
                "__py2llvm_dict_contains", ir.FunctionType(I1, [DICT_PTR, I64, I64])
            )
            res_d = self.builder.call(
                dict_cont_fn, [self.builder.inttoptr(rpay, DICT_PTR), ltag, lpay]
            )
            self.builder.branch(merge_bb)

            # STR branch — use C strstr(haystack_data, needle_data)
            self.builder.position_at_end(str_bb)
            z = ir.Constant(I32, 0)
            r_str_obj = self.builder.inttoptr(rpay, STR_PTR)
            l_str_obj = self.builder.inttoptr(lpay, STR_PTR)
            r_data = self.builder.load(
                self.builder.gep(r_str_obj, [z, ir.Constant(I32, 3)], inbounds=True)
            )
            l_data = self.builder.load(
                self.builder.gep(l_str_obj, [z, ir.Constant(I32, 3)], inbounds=True)
            )
            strstr_res = self.builder.call(self._strstr, [r_data, l_data])
            res_s = self.builder.icmp_signed(
                "!=", strstr_res, ir.Constant(I8P, None)
            )
            self.builder.branch(merge_bb)

            # OTHER branch — not a container, return false
            self.builder.position_at_end(other_bb)
            self.builder.branch(merge_bb)

            # Merge: phi from all branches
            self.builder.position_at_end(merge_bb)
            res = self.builder.phi(I1, "in_res")
            res.add_incoming(res_l, lst_bb)
            res.add_incoming(res_d, dct_bb)
            res.add_incoming(res_s, str_bb)
            res.add_incoming(ir.Constant(I1, 0), other_bb)

            if type(op) == ast.NotIn:
                res = self.builder.not_(res)
            return Value(res, PyType.BOOL)

        res = self.builder.alloca(I1, name="dcmp")
        both_int = self.builder.and_(
            self.builder.icmp_signed("==", ltag, ir.Constant(I64, Tag.INT)),
            self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.INT)),
        )
        both_str = self.builder.and_(
            self.builder.icmp_signed("==", ltag, ir.Constant(I64, Tag.STR)),
            self.builder.icmp_signed("==", rtag, ir.Constant(I64, Tag.STR)),
        )

        int_bb = self.current_func.append_basic_block("dcmp.int")
        not_int_bb = self.current_func.append_basic_block("dcmp.not_int")
        str_bb = self.current_func.append_basic_block("dcmp.str")
        flt_bb = self.current_func.append_basic_block("dcmp.flt")
        end_bb = self.current_func.append_basic_block("dcmp.end")
        self.builder.cbranch(both_int, int_bb, not_int_bb)

        ops = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        pred = ops.get(type(op))
        if pred is None:
            raise CompileError(
                f"Nieobsługiwane porównanie dynamiczne: {type(op).__name__}"
            )

        self.builder.position_at_end(int_bb)
        r = self.builder.icmp_signed(pred, lpay, rpay)
        self.builder.store(r, res)
        self.builder.branch(end_bb)

        self.builder.position_at_end(not_int_bb)
        self.builder.cbranch(both_str, str_bb, flt_bb)

        # ── STR == STR comparison using strcmp ──
        self.builder.position_at_end(str_bb)
        z = ir.Constant(I32, 0)
        l_str_obj = self.builder.inttoptr(lpay, STR_PTR, "l_str")
        r_str_obj = self.builder.inttoptr(rpay, STR_PTR, "r_str")
        l_data = self.builder.load(
            self.builder.gep(l_str_obj, [z, ir.Constant(I32, 3)], inbounds=True),
            name="l_str_data",
        )
        r_data = self.builder.load(
            self.builder.gep(r_str_obj, [z, ir.Constant(I32, 3)], inbounds=True),
            name="r_str_data",
        )
        cmp_result = self.builder.call(self._strcmp, [l_data, r_data], name="strcmp_res")
        # Map strcmp result to the comparison predicate
        # strcmp returns 0 for equal, <0 for less, >0 for greater
        if isinstance(op, ast.Eq):
            str_cmp = self.builder.icmp_signed("==", cmp_result, ir.Constant(I32, 0))
        elif isinstance(op, ast.NotEq):
            str_cmp = self.builder.icmp_signed("!=", cmp_result, ir.Constant(I32, 0))
        elif isinstance(op, ast.Lt):
            str_cmp = self.builder.icmp_signed("<", cmp_result, ir.Constant(I32, 0))
        elif isinstance(op, ast.LtE):
            str_cmp = self.builder.icmp_signed("<=", cmp_result, ir.Constant(I32, 0))
        elif isinstance(op, ast.Gt):
            str_cmp = self.builder.icmp_signed(">", cmp_result, ir.Constant(I32, 0))
        elif isinstance(op, ast.GtE):
            str_cmp = self.builder.icmp_signed(">=", cmp_result, ir.Constant(I32, 0))
        else:
            str_cmp = ir.Constant(I1, 0)
        self.builder.store(str_cmp, res)
        self.builder.branch(end_bb)

        self.builder.position_at_end(flt_bb)
        lf = self._pay_to_f64_runtime(ltag, lpay)
        rf = self._pay_to_f64_runtime(rtag, rpay)
        r = self.builder.fcmp_ordered(pred, lf, rf)
        self.builder.store(r, res)
        self.builder.branch(end_bb)

        self.builder.position_at_end(end_bb)
        return Value(self.builder.load(res), PyType.BOOL)

    # ──────────────────────────────────────────────────────────────
    #  PRINT  –  poprawna obsługa wieloargumentowa i list
    # ──────────────────────────────────────────────────────────────

    # def print_value(self, v: Value):
    #     if v.is_int:
    #         self.builder.call(self._printf, [self._str_ptr("%lld"), v.llvm])
    #     elif v.is_float:
    #         self.builder.call(self._printf, [self._str_ptr("%g"), v.llvm])
    #     elif v.is_bool:
    #         ext = self.builder.zext(v.llvm, I64)
    #         self.builder.call(self._printf, [self._str_ptr("%lld"), ext])
    #     elif v.is_str:
    #         self.builder.call(self._printf, [self._str_ptr("%s"), v.llvm])
    #     elif v.is_list:
    #         self._print_list(v)
    #     elif v.is_dict:
    #         sz = self.dict_len(v).llvm
    #         self.builder.call(self._printf,
    #                           [self._str_ptr("<dict len=%lld>"), sz])
    #     elif v.is_object:
    #         tag, pay = self._read_slot(v.llvm)
    #         self._print_tag_pay(tag, pay)
    #     else:
    #         self.builder.call(self._printf, [self._str_ptr("None")])

    def _call_llvm_function(self, func: ir.Function, node: ast.Call) -> Value:
        """Wywołuje funkcję C/LLVM (np. math.sqrt)."""
        # Get argument types from function signature
        fty = func.function_type
        arg_types = list(fty.args)

        if len(node.args) != len(arg_types):
            raise CompileError(
                f"'{func.name}' wymaga {len(arg_types)} arg, podano {len(node.args)}.",
                node,
            )

        call_args = []
        for an, et in zip(node.args, arg_types):
            v = self.visit(an)
            if et == BOXED_PTR:
                if v.is_object:
                    call_args.append(v.llvm)
                else:
                    call_args.append(self._box(v))
            else:
                v = self._cast_to_llvm(v, et, an)
                call_args.append(v.llvm)

        ret = self.builder.call(func, call_args, name=f"{func.name}_ret")

        # Return type determines PyType
        if fty.return_type == I32:
            return Value(ret, PyType.INT)
        elif fty.return_type == I64:
            return Value(ret, PyType.INT)
        elif fty.return_type == F64:
            return Value(ret, PyType.FLOAT)
        elif fty.return_type == I8P:
            return Value(ret, PyType.STRING)
        elif fty.return_type == VOID:
            return Value(ir.Constant(I64, 0), PyType.NONE)
        else:
            return Value(ret, PyType.OBJECT)

    def _can_inline(self, call_node: ast.Call) -> bool:
        """Check if function can be inlined."""
        if not call_node.args:
            return False

        for arg in call_node.args:
            if not isinstance(arg, (ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp)):
                return False

        func_name = call_node.func.id
        if func_name not in self._function_ast:
            return False

        func_ast = self._function_ast[func_name]
        stmt_count = sum(
            1
            for s in func_ast.body
            if isinstance(s, (ast.Assign, ast.AugAssign, ast.Expr, ast.Return))
        )

        return stmt_count <= self._inline_threshold

    def _inline_function(self, fname: str, call_node: ast.Call) -> Value:
        """Inline a small function at call site."""
        func_ast = self._function_ast[fname]
        arg_names = [arg.arg for arg in func_ast.args.args]

        old_sym = {}
        for param_name, arg_node in zip(arg_names, call_node.args):
            if isinstance(arg_node, ast.Constant):
                val = self.visit(arg_node)
                alloca = self.builder.alloca(I64, name=f"inline_{param_name}")
                if val.is_int:
                    self.builder.store(val.llvm, alloca)
                    old_sym[param_name] = (
                        self.sym.lookup(param_name)
                        if self.sym.exists_local(param_name)
                        else None
                    )
                    self.sym.define(param_name, VarInfo(alloca, I64, PyType.INT))
                elif val.is_float:
                    self.builder.store(val.llvm, alloca)
                    old_sym[param_name] = (
                        self.sym.lookup(param_name)
                        if self.sym.exists_local(param_name)
                        else None
                    )
                    self.sym.define(param_name, VarInfo(alloca, F64, PyType.FLOAT))

        result_val = None
        for stmt in func_ast.body:
            if isinstance(stmt, ast.Return):
                if stmt.value:
                    result_val = self.visit(stmt.value)
                break
            self.visit(stmt)

        for param_name in arg_names:
            if param_name in old_sym and old_sym[param_name]:
                self.sym.define(param_name, old_sym[param_name])
            elif self.sym.exists_local(param_name):
                pass

        return result_val if result_val else Value(ir.Constant(I64, 0), PyType.NONE)

