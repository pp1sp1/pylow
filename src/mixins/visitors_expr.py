################################################################################

"""AST visitor methods for expression nodes.

Includes FFI module reference handling: when visit_Name encounters a
variable whose VarInfo has is_ffi_module=True, it returns an
FFIModuleValue instead of attempting builder.load on a None alloca.
"""

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
from ..exceptions import CompileError, PylowError
from ..reporter import ErrorCategory, ErrorLevel
from ..symbols import VarInfo, SymbolTable
from ..values import Value, FFIModuleValue
from ..type_analyzer import StaticTypeAnalyzer

if TYPE_CHECKING:
    pass


class VisitorsExprMixin:
    """AST visitor methods for expression nodes."""

    def visit_Constant(self, node: ast.Constant) -> Value:
        v = node.value
        if isinstance(v, bool):
            return Value(ir.Constant(I1, int(v)), PyType.BOOL)
        if isinstance(v, int):
            return Value(ir.Constant(I64, v), PyType.INT)
        if isinstance(v, float):
            return Value(ir.Constant(F64, v), PyType.FLOAT)
        if isinstance(v, str):
            return self.create_string(v)
        if v is None:
            return Value(ir.Constant(I64, 0), PyType.NONE)
        raise self._error(
            ErrorCategory.UNSUPPORTED,
            f"Unsupported constant type: {type(v).__name__}",
            node,
            help_text="Pylow supports int, float, str, bool, and None constants.",
        )

    def visit_Name(self, node: ast.Name) -> Value:
        if isinstance(node.ctx, ast.Load):
            name = node.id
            # Special handling for __name__
            if name == "__name__":
                if "__name__" in self.module.globals:
                    gv = self.module.globals["__name__"]
                    z = ir.Constant(I32, 0)
                    str_ptr = self.builder.gep(gv, [z, z], inbounds=True)
                    # Create a string object for "__main__"
                    return self.create_string("__main__")
                else:
                    return self.create_string("__main__")

            # ══════════════════════════════════════════════════════════════════
            #  FFI: If this name refers to an imported native .so module,
            #  return FFIModuleValue instead of trying builder.load(None).
            #  This MUST be checked before any alloca-based load.
            # ══════════════════════════════════════════════════════════════════
            try:
                info_early = self.sym.lookup(name)
                if getattr(info_early, 'is_ffi_module', False):
                    mod_name = getattr(info_early, 'ffi_module_name', None) or name
                    return FFIModuleValue(mod_name)
            except CompileError:
                pass  # Not defined yet — fall through to normal handling

            # NAPRAWA: Check if this name is a class reference (e.g., DeviceFactory)
            # Used for ClassName.method() calls (classmethod/staticmethod)
            is_class_key = f"__is_class_{name}"
            if is_class_key in self.functions:
                # Return a class reference value — it's a special marker
                # that _method_call can use for dispatching classmethod/staticmethod calls
                return Value(ir.Constant(CLASS_PTR, None), PyType.OBJECT, class_name=name)

            # Check if this variable is declared as global and has an LLVM global
            if name in getattr(self, "_global_vars", set()):
                gvar_name = f"__global_{name}"
                if gvar_name in self.module.globals:
                    gv = self.module.globals[gvar_name]
                    loaded = self.builder.load(gv, name=name)
                    # NAPRAWA: Propaguj class_name z VarInfo dla instancji klas
                    cn = None
                    try:
                        info = self.sym.lookup(name)
                        cn = info.class_name if hasattr(info, "class_name") else None
                    except: pass
                    return Value(loaded, PyType.OBJECT, class_name=cn)
            # NAPRAWA: Jeśli zmienna jest nonlocal i ma LLVM global, czytaj z niego
            if name in getattr(self, "_nonlocal_vars", set()):
                nl_gvar_name = f"__nonlocal_{name}"
                if nl_gvar_name in self.module.globals:
                    gv = self.module.globals[nl_gvar_name]
                    loaded = self.builder.load(gv, name=name)
                    return Value(loaded, PyType.OBJECT)
            # NAPRAWA: Jeśli istnieje LLVM global __global_{name} i nie jesteśmy
            # wewnątrz funkcji (czyli _global_vars jest puste, ale zadeklarowano
            # globalną zmienną na poziomie modułu), czytaj z LLVM global zamiast
            # z lokalnego alloca.
            # WAŻNE: Używaj global fallback TYLKO gdy zmienna jest typu OBJECT (BOXED_PTR).
            # Dla prostych typów (INT, FLOAT, BOOL) używaj lokalnego alloca -
            # inaczej phi node nie może scalić BOXED_PTR z I64/F64/I1.
            gvar_name = f"__global_{name}"
            try:
                info_check = self.sym.lookup(name)
            except CompileError:
                info_check = None

            # ══════════════════════════════════════════════════════════════════
            #  FFI: Second check — if info_check is an FFI module, return
            #  FFIModuleValue before attempting any LLVM load operations.
            # ══════════════════════════════════════════════════════════════════
            if info_check is not None and getattr(info_check, 'is_ffi_module', False):
                mod_name = getattr(info_check, 'ffi_module_name', None) or name
                return FFIModuleValue(mod_name)

            use_global_fallback = (
                gvar_name in self.module.globals
                and info_check is not None
                and info_check.py_type == PyType.OBJECT
                and info_check.llvm_type == BOXED_PTR
            )

            if use_global_fallback:
                gv = self.module.globals[gvar_name]
                # Sprawdź, czy LLVM global został już zainicjalizowany (nie jest null)
                loaded = self.builder.load(gv, name=name)
                is_not_null = self.builder.icmp_signed(
                    "!=", self.builder.ptrtoint(loaded, I64), ir.Constant(I64, 0)
                )
                # Jeśli LLVM global ma wartość, użyj jej; inaczej spróbuj lokalnej
                use_global_bb = self.current_func.append_basic_block("name.gvar")
                use_local_bb = self.current_func.append_basic_block("name.local")
                merge_bb = self.current_func.append_basic_block("name.merge")
                self.builder.cbranch(is_not_null, use_global_bb, use_local_bb)

                self.builder.position_at_end(use_global_bb)
                gval = self.builder.load(gv, name=f"{name}_g")
                self.builder.branch(merge_bb)

                self.builder.position_at_end(use_local_bb)
                lval = self.builder.load(info_check.alloca, name=f"{name}_l")
                self.builder.branch(merge_bb)

                self.builder.position_at_end(merge_bb)
                phi = self.builder.phi(BOXED_PTR, f"{name}_val")
                phi.add_incoming(gval, use_global_bb)
                phi.add_incoming(lval, use_local_bb)
                cn = info_check.class_name if hasattr(info_check, 'class_name') else None
                return Value(phi, info_check.py_type, class_name=cn)

            # ══════════════════════════════════════════════════════════════════
            #  NAPRAWA: Dostęp do zmiennej modułowej BEZ deklaracji 'global'.
            #  W Pythonie czytanie zmiennej globalnej nie wymaga 'global' —
            #  to słowo kluczowe jest potrzebne tylko do zapisu.
            #  Jeśli LLVM global __global_{name} istnieje, a zmienna nie jest
            #  w symbol table (info_check is None), ładujemy z globala.
            # ══════════════════════════════════════════════════════════════════
            if info_check is None and gvar_name in self.module.globals:
                gv = self.module.globals[gvar_name]
                loaded = self.builder.load(gv, name=name)
                return Value(loaded, PyType.OBJECT)

            # NAPRAWA: Jeśli istnieje __nonlocal_{name} LLVM global (zadeklarowany
            # przez inner function), czytaj z niego – inner function mogła
            # zmienić wartość, której local alloca nie widzi.
            # WAŻNE: Ten fallback jest bezpieczny TYLKO dla OBJECT (BOXED_PTR) -
            # nonlocal LLVM globals są BOXED_PTR, a phi nie może scalić różnych typów.
            nl_gvar_name = f"__nonlocal_{name}"
            use_nonlocal_fallback = (
                nl_gvar_name in self.module.globals
                and info_check is not None
                and info_check.py_type == PyType.OBJECT
                and info_check.llvm_type == BOXED_PTR
            )
            if use_nonlocal_fallback:
                nl_gv = self.module.globals[nl_gvar_name]
                nl_loaded = self.builder.load(nl_gv, name=f"{name}_nl")
                nl_is_not_null = self.builder.icmp_signed(
                    "!=", self.builder.ptrtoint(nl_loaded, I64), ir.Constant(I64, 0)
                )
                nl_use_global_bb = self.current_func.append_basic_block("name.nl.gvar")
                nl_use_local_bb = self.current_func.append_basic_block("name.nl.local")
                nl_merge_bb = self.current_func.append_basic_block("name.nl.merge")
                self.builder.cbranch(nl_is_not_null, nl_use_global_bb, nl_use_local_bb)

                self.builder.position_at_end(nl_use_global_bb)
                nl_gval = self.builder.load(nl_gv, name=f"{name}_nl_g")
                self.builder.branch(nl_merge_bb)

                self.builder.position_at_end(nl_use_local_bb)
                nl_lval = self.builder.load(info_check.alloca, name=f"{name}_nl_l")
                self.builder.branch(nl_merge_bb)

                self.builder.position_at_end(nl_merge_bb)
                nl_phi = self.builder.phi(BOXED_PTR, f"{name}_nl_val")
                nl_phi.add_incoming(nl_gval, nl_use_global_bb)
                nl_phi.add_incoming(nl_lval, nl_use_local_bb)
                cn = info_check.class_name if hasattr(info_check, 'class_name') else None
                return Value(nl_phi, PyType.OBJECT, class_name=cn)

            # This is the main variable resolution — pass node for error context
            info = self.sym.lookup(name, node)

            # ══════════════════════════════════════════════════════════════════
            #  FFI: Final check before builder.load — if this VarInfo is an
            #  FFI module reference, return FFIModuleValue. The alloca is
            #  None for FFI modules, so builder.load would crash.
            # ══════════════════════════════════════════════════════════════════
            if getattr(info, 'is_ffi_module', False):
                mod_name = getattr(info, 'ffi_module_name', None) or name
                return FFIModuleValue(mod_name)

            # Normal variable: load from alloca
            loaded = self.builder.load(info.alloca, name=name)
            cn = info.class_name if hasattr(info, "class_name") else None
            return Value(loaded, info.py_type, class_name=cn)
        raise self._error(
            ErrorCategory.SEMANTIC,
            "Unsupported Name context (only Load is supported)",
            node,
            help_text="Variable names can only be used in a Load context in pylow.",
        )

    def visit_BinOp(self, node: ast.BinOp) -> Value:
        left = self.visit(node.left)
        right = self.visit(node.right)

        # Binary Ops Optimization: Jeśli oba operandy to unboxed INT, użyj natywnych instrukcji
        if left.is_int and right.is_int:
            if isinstance(node.op, ast.Add):
                r = self.builder.add(left.llvm, right.llvm)
            elif isinstance(node.op, ast.Sub):
                r = self.builder.sub(left.llvm, right.llvm)
            elif isinstance(node.op, ast.Mult):
                r = self.builder.mul(left.llvm, right.llvm)
            elif isinstance(node.op, ast.FloorDiv):
                self._emit_div_zero_check(right, is_float=False)
                r = self.builder.sdiv(left.llvm, right.llvm)
            elif isinstance(node.op, ast.Mod):
                self._emit_div_zero_check(right, is_float=False)
                r = self.builder.srem(left.llvm, right.llvm)
            elif isinstance(node.op, ast.BitAnd):
                r = self.builder.and_(left.llvm, right.llvm)
            elif isinstance(node.op, ast.BitOr):
                r = self.builder.or_(left.llvm, right.llvm)
            elif isinstance(node.op, ast.BitXor):
                r = self.builder.xor(left.llvm, right.llvm)
            elif isinstance(node.op, ast.LShift):
                r = self.builder.shl(left.llvm, right.llvm)
            elif isinstance(node.op, ast.RShift):
                r = self.builder.ashr(left.llvm, right.llvm)
            elif isinstance(node.op, ast.Pow):
                # int ** int - użyj wywołania funkcji pow
                pow_fn = self.functions.get("llvm.pow.f64") or self.functions.get("pow")
                if not pow_fn:
                    fty = ir.FunctionType(F64, [F64, F64])
                    pow_fn = ir.Function(self.module, fty, name="llvm.pow.f64")
                    self.functions["llvm.pow.f64"] = pow_fn
                lf = self.builder.sitofp(left.llvm, F64)
                rf = self.builder.sitofp(right.llvm, F64)
                res = self.builder.call(pow_fn, [lf, rf])
                return Value(self.builder.fptosi(res, I64), PyType.INT)
            elif isinstance(node.op, ast.Div):
                # / w Pythonie wymusza odpowiedź FLOAT
                self._emit_div_zero_check(right, is_float=False)
                lf = self.builder.sitofp(left.llvm, F64)
                rf = self.builder.sitofp(right.llvm, F64)
                return Value(self.builder.fdiv(lf, rf), PyType.FLOAT)
            else:
                raise self._error(
                    ErrorCategory.UNSUPPORTED,
                    f"Unsupported int operator: {type(node.op).__name__}",
                    node,
                    help_text="Pylow supports +, -, *, /, //, %, **, &, |, ^, <<, >> on integers.",
                )
            return Value(r, PyType.INT)

        # Fallback na istniejący silnik z coercion oraz wsparciem dla stringów i OBJECT
        return self._apply_binop(node.op, left, right, node)

    def _apply_binop(self, op, left: Value, right: Value, node=None) -> Value:
        # FIX: Obsługa formatowania stringów: "format %s" % (args) lub "format %s" % arg
        # MUSI być PRZED OBJECT check, bo zmienna po prawej stronie (np. name)
        # może być OBJECT (boxed), a my chcemy _string_format, nie dynamic_binop.
        if left.is_str and isinstance(op, ast.Mod):
            return self._string_format(left, right, node)

        if left.is_object or right.is_object:
            ltag, lpay = self._value_to_tag_payload(left)
            rtag, rpay = self._value_to_tag_payload(right)
            return self.dynamic_binop(op, ltag, lpay, rtag, rpay)

        # FIX: Obsługa formatowania stringów z TUPLE po prawej (obiekty nie-boxed).
        # MUSI być PRZED _coerce, bo _coerce próbuje konwertować STR→FLOAT co się nie uda.
        if left.is_str and isinstance(op, ast.Mod):
            return self._string_format(left, right, node)

        if left.is_str and right.is_str:
            if isinstance(op, ast.Add):
                return self.concat_strings(left, right)
            raise self._error(
                ErrorCategory.UNSUPPORTED,
                "Unsupported string operation",
                node,
                hint="Only string concatenation (+) and formatting (%) are supported",
                help_text="Use + to concatenate strings or % for formatting.",
            )

        left, right = self._coerce(left, right)
        fp = left.is_float

        # FIX: Obsługa operacji na zbiorach (listy jako zbiory): |, &, -, ^
        # Zbiory traktujemy jako listy, ale wynik oznaczamy PyType.SET (drukowany z {})
        if (left.is_list or left.is_set) and (right.is_list or right.is_set):
            if isinstance(op, ast.BitOr):
                result = self._set_union(left, right)
                return Value(result.llvm, PyType.SET)
            elif isinstance(op, ast.BitAnd):
                result = self._set_intersection(left, right)
                return Value(result.llvm, PyType.SET)
            elif isinstance(op, ast.Sub):
                result = self._set_difference(left, right)
                return Value(result.llvm, PyType.SET)
            elif isinstance(op, ast.BitXor):
                result = self._set_symmetric_difference(left, right)
                return Value(result.llvm, PyType.SET)

        if isinstance(op, ast.Add):
            if left.is_list and right.is_list:
                return self._concat_lists(left, right)
            r = (
                self.builder.fadd(left.llvm, right.llvm)
                if fp
                else self.builder.add(left.llvm, right.llvm)
            )
        elif isinstance(op, ast.Sub):
            r = (
                self.builder.fsub(left.llvm, right.llvm)
                if fp
                else self.builder.sub(left.llvm, right.llvm)
            )
        elif isinstance(op, ast.Mult):
            r = (
                self.builder.fmul(left.llvm, right.llvm)
                if fp
                else self.builder.mul(left.llvm, right.llvm)
            )
        elif isinstance(op, ast.Div):
            lf = self._to_float(left)
            rf = self._to_float(right)
            self._emit_div_zero_check(rf, is_float=True)
            return Value(self.builder.fdiv(lf.llvm, rf.llvm), PyType.FLOAT)
        elif isinstance(op, ast.FloorDiv):
            if fp:
                self._emit_div_zero_check(right, is_float=True)
                return Value(
                    self.builder.fptosi(self.builder.fdiv(left.llvm, right.llvm), I64),
                    PyType.INT,
                )
            self._emit_div_zero_check(right, is_float=False)
            r = self.builder.sdiv(left.llvm, right.llvm)
        elif isinstance(op, ast.Mod):
            if fp:
                self._emit_div_zero_check(right, is_float=True)
                r = self.builder.frem(left.llvm, right.llvm)
            else:
                self._emit_div_zero_check(right, is_float=False)
                r = self.builder.srem(left.llvm, right.llvm)
        elif isinstance(op, ast.BitAnd):
            r = self.builder.and_(left.llvm, right.llvm)
        elif isinstance(op, ast.BitOr):
            r = self.builder.or_(left.llvm, right.llvm)
        elif isinstance(op, ast.BitXor):
            r = self.builder.xor(left.llvm, right.llvm)
        elif isinstance(op, ast.LShift):
            r = self.builder.shl(left.llvm, right.llvm)
        elif isinstance(op, ast.RShift):
            r = self.builder.ashr(left.llvm, right.llvm)
        elif isinstance(op, ast.Pow):
            pow_fn = self.functions.get("llvm.pow.f64") or self.functions.get("pow")
            if not pow_fn:
                fty = ir.FunctionType(F64, [F64, F64])
                pow_fn = ir.Function(self.module, fty, name="llvm.pow.f64")
                self.functions["llvm.pow.f64"] = pow_fn
            lf = self._to_float(left).llvm
            rf = self._to_float(right).llvm
            res = self.builder.call(pow_fn, [lf, rf])
            return Value(res, PyType.FLOAT)
        else:
            raise self._error(
                ErrorCategory.UNSUPPORTED,
                f"Unsupported binary operator: {type(op).__name__}",
                node,
                help_text="Pylow supports standard arithmetic, bitwise, and comparison operators.",
            )
        return Value(r, left.pytype)

    # ──────────────────────────────────────────────────────────────
    #  Set operations on lists (treated as sets)
    # ──────────────────────────────────────────────────────────────

    def _list_contains_int(self, lst_val: Value, elem_i64: ir.Value) -> ir.Value:
        """Sprawdza, czy lista zawiera element INT (i64). Zwraca i1."""
        z = ir.Constant(I32, 0)
        sp, _, dp = self._list_ptrs(lst_val.llvm)
        sz = self.builder.load(sp, "lc_sz")
        data = self.builder.load(dp, "lc_data")
        found_a = self.builder.alloca(I1, name="lc_found")
        self.builder.store(ir.Constant(I1, 0), found_a)
        idx_a = self.builder.alloca(I64, name="lc_i")
        self.builder.store(ir.Constant(I64, 0), idx_a)
        cond_bb = self.current_func.append_basic_block("lc.cond")
        body_bb = self.current_func.append_basic_block("lc.body")
        found_bb = self.current_func.append_basic_block("lc.found")
        next_bb = self.current_func.append_basic_block("lc.next")
        end_bb = self.current_func.append_basic_block("lc.end")
        self.builder.branch(cond_bb)
        self.builder.position_at_end(cond_bb)
        ci = self.builder.load(idx_a)
        already_found = self.builder.load(found_a)
        cont = self.builder.and_(
            self.builder.icmp_signed("<", ci, sz),
            self.builder.not_(already_found)
        )
        self.builder.cbranch(cont, body_bb, end_bb)
        self.builder.position_at_end(body_bb)
        slot = self.builder.gep(data, [ci], inbounds=True)
        slot_boxed = self.builder.bitcast(slot, BOXED_PTR)
        etag = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 1)], inbounds=True))
        epay = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 2)], inbounds=True))
        is_int = self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.INT))
        val_match = self.builder.icmp_signed("==", epay, elem_i64)
        match = self.builder.and_(is_int, val_match)
        self.builder.cbranch(match, found_bb, next_bb)
        self.builder.position_at_end(found_bb)
        self.builder.store(ir.Constant(I1, 1), found_a)
        self.builder.branch(end_bb)
        self.builder.position_at_end(next_bb)
        self.builder.store(self.builder.add(ci, ir.Constant(I64, 1)), idx_a)
        self.builder.branch(cond_bb)
        self.builder.position_at_end(end_bb)
        return self.builder.load(found_a)

    def _set_union(self, left: Value, right: Value) -> Value:
        """Zbiór union: left | right."""
        z = ir.Constant(I32, 0)
        result = self._concat_lists(left, self.create_list([]))
        rsp, _, rdp = self._list_ptrs(right.llvm)
        rsz = self.builder.load(rsp, "su_rsz")
        rdata = self.builder.load(rdp, "su_rdata")
        ri_a = self.builder.alloca(I64, name="su_ri")
        self.builder.store(ir.Constant(I64, 0), ri_a)
        cond_bb = self.current_func.append_basic_block("su.cond")
        body_bb = self.current_func.append_basic_block("su.body")
        add_bb = self.current_func.append_basic_block("su.add")
        skip_bb = self.current_func.append_basic_block("su.skip")
        end_bb = self.current_func.append_basic_block("su.end")
        self.builder.branch(cond_bb)
        self.builder.position_at_end(cond_bb)
        ri = self.builder.load(ri_a)
        self.builder.cbranch(self.builder.icmp_signed("<", ri, rsz), body_bb, end_bb)
        self.builder.position_at_end(body_bb)
        slot = self.builder.gep(rdata, [ri], inbounds=True)
        slot_boxed = self.builder.bitcast(slot, BOXED_PTR)
        etag = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 1)], inbounds=True))
        epay = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 2)], inbounds=True))
        is_int_elem = self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.INT))
        found = self._list_contains_int(result, epay)
        not_found = self.builder.not_(found)
        should_add = self.builder.and_(is_int_elem, not_found)
        self.builder.cbranch(should_add, add_bb, skip_bb)
        self.builder.position_at_end(add_bb)
        elem = self._boxed_to_value(etag, epay, None)
        self.list_append(result, elem)
        self.builder.branch(skip_bb)
        self.builder.position_at_end(skip_bb)
        self.builder.store(self.builder.add(ri, ir.Constant(I64, 1)), ri_a)
        self.builder.branch(cond_bb)
        self.builder.position_at_end(end_bb)
        return result

    def _set_intersection(self, left: Value, right: Value) -> Value:
        """Zbiór intersection: left & right."""
        z = ir.Constant(I32, 0)
        result = self.create_list([])
        lsp, _, ldp = self._list_ptrs(left.llvm)
        lsz = self.builder.load(lsp, "si_lsz")
        ldata = self.builder.load(ldp, "si_ldata")
        li_a = self.builder.alloca(I64, name="si_li")
        self.builder.store(ir.Constant(I64, 0), li_a)
        cond_bb = self.current_func.append_basic_block("si.cond")
        body_bb = self.current_func.append_basic_block("si.body")
        add_bb = self.current_func.append_basic_block("si.add")
        skip_bb = self.current_func.append_basic_block("si.skip")
        end_bb = self.current_func.append_basic_block("si.end")
        self.builder.branch(cond_bb)
        self.builder.position_at_end(cond_bb)
        li = self.builder.load(li_a)
        self.builder.cbranch(self.builder.icmp_signed("<", li, lsz), body_bb, end_bb)
        self.builder.position_at_end(body_bb)
        slot = self.builder.gep(ldata, [li], inbounds=True)
        slot_boxed = self.builder.bitcast(slot, BOXED_PTR)
        etag = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 1)], inbounds=True))
        epay = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 2)], inbounds=True))
        is_int_elem = self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.INT))
        found_in_right = self._list_contains_int(right, epay)
        should_add = self.builder.and_(is_int_elem, found_in_right)
        self.builder.cbranch(should_add, add_bb, skip_bb)
        self.builder.position_at_end(add_bb)
        elem = self._boxed_to_value(etag, epay, None)
        self.list_append(result, elem)
        self.builder.branch(skip_bb)
        self.builder.position_at_end(skip_bb)
        self.builder.store(self.builder.add(li, ir.Constant(I64, 1)), li_a)
        self.builder.branch(cond_bb)
        self.builder.position_at_end(end_bb)
        return result

    def _set_difference(self, left: Value, right: Value) -> Value:
        """Zbiór difference: left - right."""
        z = ir.Constant(I32, 0)
        result = self.create_list([])
        lsp, _, ldp = self._list_ptrs(left.llvm)
        lsz = self.builder.load(lsp, "sd_lsz")
        ldata = self.builder.load(ldp, "sd_ldata")
        li_a = self.builder.alloca(I64, name="sd_li")
        self.builder.store(ir.Constant(I64, 0), li_a)
        cond_bb = self.current_func.append_basic_block("sd.cond")
        body_bb = self.current_func.append_basic_block("sd.body")
        add_bb = self.current_func.append_basic_block("sd.add")
        skip_bb = self.current_func.append_basic_block("sd.skip")
        end_bb = self.current_func.append_basic_block("sd.end")
        self.builder.branch(cond_bb)
        self.builder.position_at_end(cond_bb)
        li = self.builder.load(li_a)
        self.builder.cbranch(self.builder.icmp_signed("<", li, lsz), body_bb, end_bb)
        self.builder.position_at_end(body_bb)
        slot = self.builder.gep(ldata, [li], inbounds=True)
        slot_boxed = self.builder.bitcast(slot, BOXED_PTR)
        etag = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 1)], inbounds=True))
        epay = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 2)], inbounds=True))
        is_int_elem = self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.INT))
        found_in_right = self._list_contains_int(right, epay)
        not_found = self.builder.not_(found_in_right)
        should_add = self.builder.and_(is_int_elem, not_found)
        self.builder.cbranch(should_add, add_bb, skip_bb)
        self.builder.position_at_end(add_bb)
        elem = self._boxed_to_value(etag, epay, None)
        self.list_append(result, elem)
        self.builder.branch(skip_bb)
        self.builder.position_at_end(skip_bb)
        self.builder.store(self.builder.add(li, ir.Constant(I64, 1)), li_a)
        self.builder.branch(cond_bb)
        self.builder.position_at_end(end_bb)
        return result

    def _set_symmetric_difference(self, left: Value, right: Value) -> Value:
        """Zbiór symmetric difference: left ^ right."""
        diff1 = self._set_difference(left, right)
        diff2 = self._set_difference(right, left)
        return self._set_union(diff1, diff2)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Value:
        v = self.visit(node.operand)
        op = node.op
        if isinstance(op, ast.USub):
            if v.is_float:
                return Value(
                    self.builder.fsub(ir.Constant(F64, 0.0), v.llvm), PyType.FLOAT
                )
            if v.is_object:
                tag, pay = self._read_slot(v.llvm)
                int_bb = self.current_func.append_basic_block("uneg.int")
                flt_bb = self.current_func.append_basic_block("uneg.flt")
                err_bb = self.current_func.append_basic_block("uneg.err")
                end_bb = self.current_func.append_basic_block("uneg.end")
                res = self.builder.alloca(I64, name="uneg_res")
                res_tag = self.builder.alloca(I64, name="uneg_tag")
                self.builder.store(ir.Constant(I64, Tag.INT), res_tag)
                sw = self.builder.switch(tag, err_bb)
                sw.add_case(ir.Constant(I64, Tag.INT), int_bb)
                sw.add_case(ir.Constant(I64, Tag.FLOAT), flt_bb)

                self.builder.position_at_end(int_bb)
                self.builder.store(self.builder.neg(pay), res)
                self.builder.branch(end_bb)

                self.builder.position_at_end(flt_bb)
                fv = self.builder.bitcast(pay, F64)
                neg_fv = self.builder.fsub(ir.Constant(F64, 0.0), fv)
                self.builder.store(self.builder.bitcast(neg_fv, I64), res)
                self.builder.store(ir.Constant(I64, Tag.FLOAT), res_tag)
                self.builder.branch(end_bb)

                self.builder.position_at_end(err_bb)
                self.builder.store(ir.Constant(I64, 0), res)
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
                    null_i8p,
                    self.builder.gep(bv, [z, z, ir.Constant(I32, 3)], inbounds=True),
                )

                self.builder.store(
                    self.builder.load(res_tag),
                    self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True),
                )
                self.builder.store(
                    self.builder.load(res),
                    self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True),
                )
                return Value(bv, PyType.OBJECT)
            iv = self._to_int(v)
            return Value(self.builder.neg(iv.llvm), PyType.INT)
        if isinstance(op, ast.UAdd):
            return v
        if isinstance(op, ast.Invert):
            return Value(self.builder.not_(v.llvm), v.pytype)
        if isinstance(op, ast.Not):
            bv = self._to_bool_val(v)
            return Value(self.builder.not_(bv.llvm), PyType.BOOL)
        raise self._error(
            ErrorCategory.UNSUPPORTED,
            f"Unsupported unary operator: {type(op).__name__}",
            node,
            help_text="Pylow supports -, +, ~, and not unary operators.",
        )

    # ══════════════════════════════════════════════════════════════════
    #  POPRAWKA 2: Short-circuit evaluation zwracający oryginalne wartości (Test 10)
    # ══════════════════════════════════════════════════════════════════

    def visit_BoolOp(self, node: ast.BoolOp) -> Value:
        if isinstance(node.op, ast.Or):
            return self._compile_or_chain(node.values, node)
        else:
            return self._compile_and_chain(node.values, node)

    def _compile_or_chain(self, values: list, node: ast.AST) -> Value:
        func = self.current_func
        end_block = func.append_basic_block("or.end")
        result_alloca = self.builder.alloca(BOXED_TY, name="or_result")

        for i, val_node in enumerate(values):
            is_last = i == len(values) - 1
            current_val = self.visit(val_node)
            is_truthy = self._eval_truthiness(current_val)

            if is_last:
                self._store_value_to_alloca(result_alloca, current_val)
                self.builder.branch(end_block)
            else:
                true_bb = func.append_basic_block(f"or.true_{i}")
                next_bb = func.append_basic_block(f"or.next_{i}")
                self.builder.cbranch(is_truthy.llvm, true_bb, next_bb)
                self.builder.position_at_end(true_bb)
                self._store_value_to_alloca(result_alloca, current_val)
                self.builder.branch(end_block)
                self.builder.position_at_end(next_bb)

        self.builder.position_at_end(end_block)
        return self._load_value_from_alloca(result_alloca, node)

    def _compile_and_chain(self, values: list, node: ast.AST) -> Value:
        func = self.current_func
        end_block = func.append_basic_block("and.end")
        result_alloca = self.builder.alloca(BOXED_TY, name="and_result")

        for i, val_node in enumerate(values):
            is_last = i == len(values) - 1
            current_val = self.visit(val_node)
            is_truthy = self._eval_truthiness(current_val)
            is_falsy = self.builder.not_(is_truthy.llvm)

            if is_last:
                self._store_value_to_alloca(result_alloca, current_val)
                self.builder.branch(end_block)
            else:
                false_bb = func.append_basic_block(f"and.false_{i}")
                next_bb = func.append_basic_block(f"and.next_{i}")
                self.builder.cbranch(is_falsy, false_bb, next_bb)
                self.builder.position_at_end(false_bb)
                self._store_value_to_alloca(result_alloca, current_val)
                self.builder.branch(end_block)
                self.builder.position_at_end(next_bb)

        self.builder.position_at_end(end_block)
        return self._load_value_from_alloca(result_alloca, node)

    def _eval_truthiness(self, val: Value) -> Value:
        z = ir.Constant(I32, 0)

        if val.is_none:
            return Value(ir.Constant(I1, False), PyType.BOOL)

        if val.is_bool:
            return val

        if val.is_int:
            return Value(
                self.builder.icmp_signed("!=", val.llvm, ir.Constant(I64, 0)),
                PyType.BOOL,
            )

        if val.is_float:
            return Value(
                self.builder.fcmp_ordered("!=", val.llvm, ir.Constant(F64, 0.0)),
                PyType.BOOL,
            )

        if val.is_str:
            str_ptr = val.llvm
            len_ptr = self.builder.gep(
                str_ptr, [z, ir.Constant(I32, 1)], inbounds=True
            )
            str_len = self.builder.load(len_ptr, "str_len")
            return Value(
                self.builder.icmp_signed("!=", str_len, ir.Constant(I64, 0)),
                PyType.BOOL,
            )

        if val.is_list:
            list_ptr = val.llvm
            size_ptr = self.builder.gep(
                list_ptr, [z, ir.Constant(I32, 1)], inbounds=True
            )
            list_size = self.builder.load(size_ptr, "list_size")
            return Value(
                self.builder.icmp_signed("!=", list_size, ir.Constant(I64, 0)),
                PyType.BOOL,
            )

        if val.is_dict:
            dict_ptr = val.llvm
            size_ptr = self.builder.gep(
                dict_ptr, [z, ir.Constant(I32, 1)], inbounds=True
            )
            dict_size = self.builder.load(size_ptr, "dict_size")
            return Value(
                self.builder.icmp_signed("!=", dict_size, ir.Constant(I64, 0)),
                PyType.BOOL,
            )

        if val.is_object:
            boxed = val.llvm
            tag_ptr = self.builder.gep(boxed, [z, ir.Constant(I32, 1)], inbounds=True)
            tag = self.builder.load(tag_ptr, "box_tag")

            pay_ptr = self.builder.gep(boxed, [z, ir.Constant(I32, 2)], inbounds=True)
            pay = self.builder.load(pay_ptr, "box_pay")

            func = self.current_func
            none_bb = func.append_basic_block("truthy.none")
            int_bb = func.append_basic_block("truthy.int")
            bool_bb = func.append_basic_block("truthy.bool")
            float_bb = func.append_basic_block("truthy.float")
            str_bb = func.append_basic_block("truthy.str")
            list_bb = func.append_basic_block("truthy.list")
            tuple_bb = func.append_basic_block("truthy.tuple")
            dict_bb = func.append_basic_block("truthy.dict")
            other_bb = func.append_basic_block("truthy.other")
            merge_bb = func.append_basic_block("truthy.merge")

            sw = self.builder.switch(tag, other_bb)
            sw.add_case(ir.Constant(I64, Tag.NONE), none_bb)
            sw.add_case(ir.Constant(I64, Tag.INT), int_bb)
            sw.add_case(ir.Constant(I64, Tag.BOOL), bool_bb)
            sw.add_case(ir.Constant(I64, Tag.FLOAT), float_bb)
            sw.add_case(ir.Constant(I64, Tag.STR), str_bb)
            sw.add_case(ir.Constant(I64, Tag.LIST), list_bb)
            sw.add_case(ir.Constant(I64, Tag.TUPLE), tuple_bb)
            sw.add_case(ir.Constant(I64, Tag.SET), list_bb)
            sw.add_case(ir.Constant(I64, Tag.DICT), dict_bb)

            self.builder.position_at_end(none_bb)
            self.builder.branch(merge_bb)

            self.builder.position_at_end(int_bb)
            int_truthy = self.builder.icmp_signed("!=", pay, ir.Constant(I64, 0))
            self.builder.branch(merge_bb)

            self.builder.position_at_end(bool_bb)
            bool_truthy = self.builder.icmp_signed("!=", pay, ir.Constant(I64, 0))
            self.builder.branch(merge_bb)

            self.builder.position_at_end(float_bb)
            float_val = self.builder.bitcast(pay, F64)
            float_truthy = self.builder.fcmp_ordered("!=", float_val, ir.Constant(F64, 0.0))
            self.builder.branch(merge_bb)

            self.builder.position_at_end(str_bb)
            str_ptr = self.builder.inttoptr(pay, STR_PTR)
            str_len_ptr = self.builder.gep(str_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
            str_len = self.builder.load(str_len_ptr, "str_len")
            str_truthy = self.builder.icmp_signed("!=", str_len, ir.Constant(I64, 0))
            self.builder.branch(merge_bb)

            self.builder.position_at_end(list_bb)
            list_ptr = self.builder.inttoptr(pay, LIST_PTR)
            list_size_ptr = self.builder.gep(list_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
            list_size = self.builder.load(list_size_ptr, "list_size")
            list_truthy = self.builder.icmp_signed("!=", list_size, ir.Constant(I64, 0))
            self.builder.branch(merge_bb)

            self.builder.position_at_end(tuple_bb)
            tuple_ptr = self.builder.inttoptr(pay, LIST_PTR)
            tuple_size_ptr = self.builder.gep(tuple_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
            tuple_size = self.builder.load(tuple_size_ptr, "tuple_size")
            tuple_truthy = self.builder.icmp_signed("!=", tuple_size, ir.Constant(I64, 0))
            self.builder.branch(merge_bb)

            self.builder.position_at_end(dict_bb)
            dict_ptr = self.builder.inttoptr(pay, DICT_PTR)
            dict_size_ptr = self.builder.gep(dict_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
            dict_size = self.builder.load(dict_size_ptr, "dict_size")
            dict_truthy = self.builder.icmp_signed("!=", dict_size, ir.Constant(I64, 0))
            self.builder.branch(merge_bb)

            self.builder.position_at_end(other_bb)
            self.builder.branch(merge_bb)

            self.builder.position_at_end(merge_bb)
            result = self.builder.phi(I1, "truthy_result")

            result.add_incoming(ir.Constant(I1, False), none_bb)
            result.add_incoming(int_truthy, int_bb)
            result.add_incoming(bool_truthy, bool_bb)
            result.add_incoming(float_truthy, float_bb)
            result.add_incoming(str_truthy, str_bb)
            result.add_incoming(list_truthy, list_bb)
            result.add_incoming(tuple_truthy, tuple_bb)
            result.add_incoming(dict_truthy, dict_bb)
            result.add_incoming(ir.Constant(I1, True), other_bb)

            return Value(result, PyType.BOOL)

        # Fallback - assume truthy
        return Value(ir.Constant(I1, True), PyType.BOOL)

    def _store_value_to_alloca(self, alloca: ir.AllocaInstr, val: Value) -> None:
        z = ir.Constant(I32, 0)
        tag, pay = self._value_to_tag_payload(val)
        self.builder.call(
            self._memset,
            [
                self.builder.bitcast(alloca, I8P),
                ir.Constant(I32, 0),
                ir.Constant(I64, SZ_BOXED),
            ],
        )
        tag_ptr = self.builder.gep(alloca, [z, ir.Constant(I32, 1)], inbounds=True)
        pay_ptr = self.builder.gep(alloca, [z, ir.Constant(I32, 2)], inbounds=True)
        self.builder.store(tag, tag_ptr)
        self.builder.store(pay, pay_ptr)

    def _load_value_from_alloca(self, alloca: ir.AllocaInstr, node: ast.AST) -> Value:
        z = ir.Constant(I32, 0)
        boxed_result = self.builder.alloca(BOXED_TY, name="final_boxed")
        self.builder.store(self.builder.load(alloca), boxed_result)
        return Value(boxed_result, PyType.OBJECT)

    def visit_Compare(self, node: ast.Compare) -> Value:
        if len(node.ops) == 1:
            l = self.visit(node.left)
            r = self.visit(node.comparators[0])
            return self._compare_two(node.ops[0], l, r, node)
        parts = []
        left = self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            right = self.visit(comp)
            parts.append(self._compare_two(op, left, right, node))
            left = right
        result = parts[0].llvm
        for p in parts[1:]:
            result = self.builder.and_(result, p.llvm)
        return Value(result, PyType.BOOL)

    def _compare_two(self, op, left: Value, right: Value, node=None) -> Value:
        # In/NotIn always need dynamic dispatch (container membership)
        if isinstance(op, (ast.In, ast.NotIn)):
            ltag, lpay = self._value_to_tag_payload(left)
            rtag, rpay = self._value_to_tag_payload(right)
            return self.dynamic_compare(op, ltag, lpay, rtag, rpay)

        if left.is_object or right.is_object:
            ltag, lpay = self._value_to_tag_payload(left)
            rtag, rpay = self._value_to_tag_payload(right)
            return self.dynamic_compare(op, ltag, lpay, rtag, rpay)

        left, right = self._coerce(left, right)
        fp = left.is_float
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
            raise self._error(
                ErrorCategory.UNSUPPORTED,
                f"Unsupported comparison operator: {type(op).__name__}",
                node,
                help_text="Pylow supports ==, !=, <, <=, >, >=, in, not in, is, and is not.",
            )
        r = (
            self.builder.fcmp_ordered(pred, left.llvm, right.llvm)
            if fp
            else self.builder.icmp_signed(pred, left.llvm, right.llvm)
        )
        return Value(r, PyType.BOOL)

    def visit_List(self, node: ast.List) -> Value:
        return self.create_list([self.visit(e) for e in node.elts])

    def visit_Dict(self, node: ast.Dict) -> Value:
        pairs = [(self.visit(k), self.visit(v)) for k, v in zip(node.keys, node.values)]
        return self.create_dict(pairs)

    def visit_Tuple(self, node: ast.Tuple) -> Value:
        return self.create_tuple([self.visit(e) for e in node.elts])

    def visit_Set(self, node: ast.Set) -> Value:
        seen = set()
        unique_elems = []
        for e in node.elts:
            if isinstance(e, ast.Constant):
                key = (type(e.value), e.value)
                if key not in seen:
                    seen.add(key)
                    unique_elems.append(self.visit(e))
            else:
                unique_elems.append(self.visit(e))
        lst_val = self.create_list(unique_elems)
        return Value(lst_val.llvm, PyType.SET)

    # ══════════════════════════════════════════════════════════════════
    #  ASYNC/AWAIT — Obsługa coroutine
    # ══════════════════════════════════════════════════════════════════

    def visit_Await(self, node: ast.Await) -> Value:
        """Obsługa wyrażenia await — prawdziwa asynchroniczność.

        Sprawdza, czy wartość jest Task (Tag.TASK w boxed value).
        Jeśli tak, oddaje sterowanie do schedulera i czeka na zakończenie
        zadania.  Jeśli nie (np. None z asyncio.sleep()), po prostu ją zwraca.

        Uwaga: Task handles mają PyType.OBJECT z Tag.TASK, nie PyType.TASK,
        aby współpracować z systemem przypisań zmiennych.
        """
        val = self.visit(node.value)

        # Task handles are PyType.OBJECT with Tag.TASK at runtime.
        # We need to check the tag at runtime to determine if this is a task.
        if val.is_object and val.llvm.type == BOXED_PTR:
            tag, pay = self._read_slot(val.llvm)
            is_task_tag = self.builder.icmp_signed(
                "==", tag, ir.Constant(I64, Tag.TASK)
            )

            task_bb = self.current_func.append_basic_block("await.task")
            no_task_bb = self.current_func.append_basic_block("await.no_task")
            merge_bb = self.current_func.append_basic_block("await.merge")

            self.builder.cbranch(is_task_tag, task_bb, no_task_bb)

            # Task path: await and get result
            self.builder.position_at_end(task_bb)
            task_ptr = self.builder.inttoptr(pay, I8P, "await_task_ptr")
            self.builder.call(
                self.functions["__async_await_task"],
                [task_ptr]
            )
            task_result = self.builder.call(
                self.functions["__async_task_result"],
                [task_ptr],
                name="await_result"
            )
            self.builder.branch(merge_bb)

            # Non-task path: just pass through
            self.builder.position_at_end(no_task_bb)
            no_task_result = val.llvm
            self.builder.branch(merge_bb)

            # Merge
            self.builder.position_at_end(merge_bb)
            phi = self.builder.phi(BOXED_PTR, "await_phi")
            phi.add_incoming(task_result, task_bb)
            phi.add_incoming(no_task_result, no_task_bb)
            return Value(phi, PyType.OBJECT)

        # Not a boxed value — just return it (e.g. None from sleep)
        return val
