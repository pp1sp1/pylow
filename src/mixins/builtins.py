"""Built-in Python function implementations (all, any, sum, etc.)."""

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


class BuiltinsMixin:
    """Built-in Python function implementations (all, any, sum, etc.)."""

    def _handle_builtin_all(self, args: list, node: ast.Call) -> Value:
        """
        all(iterable) -> bool
        Zwraca True jesli wszystkie elementy sa truthy.
        """
        if len(args) != 1:
            raise CompileError("all() wymaga dokladnie jednego argumentu", node)

        iterable = args[0]

        if not (iterable.is_list or iterable.is_tuple or iterable.is_object):
            raise CompileError("all() wymaga argumentu typu list lub object", node)

        z = ir.Constant(I32, 0)
        func = self.current_func

        # Alloca musi byc przed jakimkolwiek branch (w biezacym bloku entry)
        idx_alloca = self.builder.alloca(I64, name="all_idx")
        result_alloca = self.builder.alloca(I1, name="all_result")
        self.builder.store(ir.Constant(I64, 0), idx_alloca)
        self.builder.store(ir.Constant(I1, 1), result_alloca)

        if iterable.is_list or iterable.is_tuple:
            sp, _, dp = self._list_ptrs(iterable.llvm)
            list_size = self.builder.load(sp, "all_size")
            list_data = self.builder.load(dp, "all_data")
        elif iterable.is_object:
            tag, pay = self._read_slot(iterable.llvm)
            pay_ptr_generic = self.builder.inttoptr(pay, LIST_PTR)
            list_size = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 1)], inbounds=True), "all_size")
            list_data = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 3)], inbounds=True), "all_data")
        else:
            raise CompileError("all() wymaga argumentu typu list lub object", node)

        loop_cond = func.append_basic_block("all.cond")
        loop_body = func.append_basic_block("all.body")
        loop_false = func.append_basic_block("all.false")
        loop_end = func.append_basic_block("all.end")

        self.builder.branch(loop_cond)

        self.builder.position_at_end(loop_cond)
        idx = self.builder.load(idx_alloca, "all_idx")
        self.builder.cbranch(
            self.builder.icmp_signed("<", idx, list_size), loop_body, loop_end
        )

        self.builder.position_at_end(loop_body)

        elem_ptr = self.builder.gep(list_data, [idx], inbounds=True)
        elem_tag = self.builder.load(
            self.builder.gep(elem_ptr, [z, ir.Constant(I32, 1)], inbounds=True),
            "all_etag",
        )
        elem_pay = self.builder.load(
            self.builder.gep(elem_ptr, [z, ir.Constant(I32, 2)], inbounds=True),
            "all_epay",
        )

        elem_val = self._boxed_to_value(elem_tag, elem_pay, node)
        is_truthy = self._eval_truthiness(elem_val)

        continue_bb = func.append_basic_block("all.continue")
        self.builder.cbranch(is_truthy.llvm, continue_bb, loop_false)

        self.builder.position_at_end(continue_bb)
        self.builder.store(self.builder.add(idx, ir.Constant(I64, 1)), idx_alloca)
        self.builder.branch(loop_cond)

        self.builder.position_at_end(loop_false)
        self.builder.store(ir.Constant(I1, 0), result_alloca)
        self.builder.branch(loop_end)

        self.builder.position_at_end(loop_end)
        result_i1 = self.builder.load(result_alloca, "all_res")
        return Value(result_i1, PyType.BOOL)

    def _handle_builtin_any(self, args: list, node: ast.Call) -> Value:
        """any(iterable) -> bool"""
        if len(args) != 1:
            raise CompileError("any() wymaga dokladnie jednego argumentu", node)

        iterable = args[0]
        if not (iterable.is_list or iterable.is_tuple or iterable.is_object):
            raise CompileError("any() wymaga argumentu typu list lub object", node)

        z = ir.Constant(I32, 0)
        func = self.current_func

        # Alloca result
        result_alloca = self.builder.alloca(I1, name="any_result")
        self.builder.store(ir.Constant(I1, 0), result_alloca)
        idx_alloca = self.builder.alloca(I64, name="any_idx")
        self.builder.store(ir.Constant(I64, 0), idx_alloca)

        if iterable.is_list or iterable.is_tuple:
            sp, _, dp = self._list_ptrs(iterable.llvm)
            list_size = self.builder.load(sp, "any_size")
            list_data = self.builder.load(dp, "any_data")
        elif iterable.is_object:
            tag, pay = self._read_slot(iterable.llvm)
            pay_ptr_generic = self.builder.inttoptr(pay, LIST_PTR)
            list_size = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 1)], inbounds=True), "any_size")
            list_data = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 3)], inbounds=True), "any_data")
        else:
            raise CompileError("any() wymaga argumentu typu list lub object", node)

        loop_start = func.append_basic_block("any.start")
        cond_bb = func.append_basic_block("any.cond")
        body_bb = func.append_basic_block("any.body")
        end_bb = func.append_basic_block("any.end")

        self.builder.branch(loop_start)

        self.builder.position_at_end(loop_start)
        self.builder.branch(cond_bb)

        self.builder.position_at_end(cond_bb)
        idx = self.builder.load(idx_alloca, "any_idx")
        self.builder.cbranch(
            self.builder.icmp_signed("<", idx, list_size), body_bb, end_bb
        )

        self.builder.position_at_end(body_bb)
        # Evaluate element truthiness
        elem_ptr = self.builder.gep(list_data, [idx], inbounds=True)
        elem_ptr_boxed = self.builder.bitcast(elem_ptr, BOXED_PTR)
        elem_tag = self.builder.load(
            self.builder.gep(elem_ptr_boxed, [z, ir.Constant(I32, 1)], inbounds=True),
            "any_etag",
        )
        elem_pay = self.builder.load(
            self.builder.gep(elem_ptr_boxed, [z, ir.Constant(I32, 2)], inbounds=True),
            "any_epay",
        )
        elem_val = self._boxed_to_value(elem_tag, elem_pay, node)
        is_truthy = self._eval_truthiness(elem_val)
        # NAPRAWA: Short-circuit – jeśli truthy, zapisz True i skocz do end
        true_exit_bb = func.append_basic_block("any.true_exit")
        continue_bb = func.append_basic_block("any.continue")
        self.builder.cbranch(is_truthy.llvm, true_exit_bb, continue_bb)

        self.builder.position_at_end(true_exit_bb)
        self.builder.store(ir.Constant(I1, 1), result_alloca)
        self.builder.branch(end_bb)

        self.builder.position_at_end(continue_bb)
        self.builder.store(self.builder.add(idx, ir.Constant(I64, 1)), idx_alloca)
        self.builder.branch(cond_bb)

        self.builder.position_at_end(end_bb)
        result_i1 = self.builder.load(result_alloca, "any_res")
        return Value(result_i1, PyType.BOOL)

    def _handle_builtin_sum(self, args: list, node: ast.Call) -> Value:
        """sum(iterable[, start]) -> number"""
        if not args:
            raise CompileError("sum() wymaga argumentu", node)

        iterable = args[0]
        start_val = args[1] if len(args) > 1 else Value(ir.Constant(I64, 0), PyType.INT)

        if not iterable.is_list:
            raise CompileError("sum() wymaga argumentu typu list", node)

        # Uproszczenie: assume all elements are int
        z = ir.Constant(I32, 0)
        func = self.current_func

        list_ptr = iterable.llvm
        size = self.builder.load(
            self.builder.gep(list_ptr, [z, z, ir.Constant(I32, 1)], inbounds=True)
        )
        data = self.builder.load(
            self.builder.gep(list_ptr, [z, z, ir.Constant(I32, 3)], inbounds=True)
        )

        result_alloca = self.builder.alloca(I64, name="sum_result")
        self.builder.store(start_val.llvm, result_alloca)

        idx_alloca = self.builder.alloca(I64, name="sum_idx")
        self.builder.store(ir.Constant(I64, 0), idx_alloca)

        cond_bb = func.append_basic_block("sum.cond")
        body_bb = func.append_basic_block("sum.body")
        end_bb = func.append_basic_block("sum.end")

        self.builder.branch(cond_bb)

        self.builder.position_at_end(cond_bb)
        idx = self.builder.load(idx_alloca)
        self.builder.cbranch(self.builder.icmp_signed("<", idx, size), body_bb, end_bb)

        self.builder.position_at_end(body_bb)
        elem_ptr = self.builder.gep(data, [idx], inbounds=True)
        elem_pay = self.builder.load(
            self.builder.gep(elem_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
        )

        curr = self.builder.load(result_alloca)
        new_val = self.builder.add(curr, elem_pay, "sum_add")
        self.builder.store(new_val, result_alloca)

        self.builder.store(self.builder.add(idx, ir.Constant(I64, 1)), idx_alloca)
        self.builder.branch(cond_bb)

        self.builder.position_at_end(end_bb)
        return Value(self.builder.load(result_alloca), PyType.INT)

    def _handle_builtin_sorted(self, args: list, node: ast.Call) -> Value:
        """sorted(iterable) -> list - uproszczenie: zwróć kopię"""
        if not args or not args[0].is_list:
            raise CompileError("sorted() wymaga argumentu typu list", node)
        # Uproszczenie: zwróć tę samą listę (bez sortowania)
        return args[0]

    def _handle_builtin_isinstance(self, args: list, node: ast.Call) -> Value:
        """isinstance(obj, class_or_tuple) -> bool"""
        # Uproszczona implementacja - zawsze zwraca False
        # Pełna implementacja wymaga systemu typów runtime
        return Value(ir.Constant(I1, False), PyType.BOOL)

    def _handle_builtin_chr(self, args: list, node: ast.Call) -> Value:
        """chr(i) -> str"""
        if len(args) != 1 or not args[0].is_int:
            raise CompileError("chr() wymaga jednego argumentu int", node)

        # Alokuj bufor na 1 znak + null terminator
        buf = self.builder.call(self._malloc, [ir.Constant(I64, 2)], "chr_buf")
        self.builder.store(args[0].llvm, buf)  # Store char code as byte
        null_pos = self.builder.gep(buf, [ir.Constant(I64, 1)], inbounds=True)
        self.builder.store(ir.Constant(I8, 0), null_pos)

        return self._create_str_object(buf, ir.Constant(I64, 1), ir.Constant(I64, 2))

    def _handle_builtin_ord(self, args: list, node: ast.Call) -> Value:
        """ord(c) -> int"""
        z = ir.Constant(I32, 0)

        if len(args) != 1 or not args[0].is_str:
            raise CompileError("ord() wymaga jednego argumentu str", node)

        str_ptr = args[0].llvm
        data_ptr = self.builder.load(
            self.builder.gep(str_ptr, [z, z, ir.Constant(I32, 3)], inbounds=True)
        )
        first_char = self.builder.load(data_ptr, "first_char")
        result = self.builder.zext(first_char, I64, "ord_result")

        return Value(result, PyType.INT)

    def _handle_builtin_id(self, args: list, node: ast.Call) -> Value:
        """id(obj) -> int (adres pamięci)"""
        if not args:
            raise CompileError("id() wymaga argumentu", node)

        val = args[0]
        if val.is_object:
            ptr = val.llvm
        else:
            ptr = self.builder.alloca(pytype_to_llvm(val.pytype))
            self.builder.store(val.llvm, ptr)

        addr = self.builder.ptrtoint(ptr, I64, "obj_id")
        return Value(addr, PyType.INT)

    def _print_string(self, str_val: Value):
        """Drukuje napis (Value o pytype STR) bez znaku nowej linii."""
        z = ir.Constant(I32, 0)
        # STR_TY = { GC_HEADER, len, cap, data_ptr }
        # str_val.llvm to STR_PTR – pobierz wskaźnik danych (index 3)
        str_obj = str_val.llvm
        data_ptr = self.builder.load(
            self.builder.gep(str_obj, [z, ir.Constant(I32, 3)], inbounds=True),
            "prompt_data"
        )
        self.builder.call(self._printf, [self._str_ptr("%s"), data_ptr])

    def _handle_builtin_input(self, args: list, node: ast.Call) -> Value:
        """input([prompt]) -> str"""
        z = ir.Constant(I32, 0)

        # Zadeklaruj fgets
        if "fgets" not in self.functions:
            fty = ir.FunctionType(I8P, [I8P, I32, I8P])
            fn = ir.Function(self.module, fty, name="fgets")
            self.functions["fgets"] = fn

        if "stdin" not in self.module.globals:
            stdin_ty = ir.LiteralStructType([])
            stdin = ir.GlobalVariable(self.module, stdin_ty, "stdin")
            stdin.linkage = "external"

        # Bufor na input
        buf_size = 1024
        buf = self.builder.call(self._malloc, [ir.Constant(I64, buf_size)], "input_buf")

        # Wyświetl prompt jeśli podany
        if args:
            prompt = self.val_to_str(args[0])
            self._print_string(prompt)

        # Wywołaj fgets
        stdin_ptr = self.module.globals["stdin"]
        stdin_i8p = self.builder.bitcast(stdin_ptr, I8P)
        self.builder.call(
            self.functions["fgets"], [buf, ir.Constant(I32, buf_size), stdin_i8p]
        )

        # Oblicz długość
        len_val = (
            self.builder.call(self.functions["strlen"], [buf], "input_len")
            if "strlen" in self.functions
            else ir.Constant(I64, 0)
        )

        return self._create_str_object(buf, len_val, ir.Constant(I64, buf_size))

    def _handle_builtin_zip(self, node: ast.Call) -> Value:
        if not node.args:
            return self.create_list([])
        iterables = [self.visit(a) for a in node.args]
        z = ir.Constant(I32, 0)
        func = self.current_func
        n_lists = len(iterables)
        list_sizes = []
        list_datas = []

        for it in iterables:
            if it.is_list:
                # Bezpośredni dostęp do listy – LIST_TY = {GC_HEADER, size, cap, data_ptr}
                sp, _, dp = self._list_ptrs(it.llvm)
                sz = self.builder.load(sp, "zip_sz")
                dt = self.builder.load(dp, "zip_dt")
                list_sizes.append(sz)
                list_datas.append(dt)
            elif it.is_object:
                # BOXED: odczytaj tag i payload, a potem rozpakuj
                tag, pay = self._read_slot(it.llvm)
                pay_ptr_generic = self.builder.inttoptr(pay, LIST_PTR)
                sz = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 1)], inbounds=True), "zip_sz")
                dt = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 3)], inbounds=True), "zip_dt")
                list_sizes.append(sz)
                list_datas.append(dt)
            else:
                raise CompileError("zip() wymaga argumentow iterowalnych (list lub object)", node)

        min_size = list_sizes[0]
        for sz in list_sizes[1:]:
            min_size = self.builder.select(self.builder.icmp_signed("<", sz, min_size), sz, min_size)
        result = self.create_list([])
        idx_a = self.builder.alloca(I64, name="zip_idx")
        self.builder.store(ir.Constant(I64, 0), idx_a)
        cond_bb = func.append_basic_block("zip.cond")
        body_bb = func.append_basic_block("zip.body")
        end_bb = func.append_basic_block("zip.end")
        self.builder.branch(cond_bb)
        self.builder.position_at_end(cond_bb)
        ci = self.builder.load(idx_a)
        self.builder.cbranch(self.builder.icmp_signed("<", ci, min_size), body_bb, end_bb)
        self.builder.position_at_end(body_bb)
        tuple_elem = self.create_tuple([])
        for li in range(n_lists):
            elem_ptr = self.builder.gep(list_datas[li], [ci], inbounds=True)
            elem_ptr_boxed = self.builder.bitcast(elem_ptr, BOXED_PTR)
            etag = self.builder.load(self.builder.gep(elem_ptr_boxed, [z, ir.Constant(I32, 1)], inbounds=True))
            epay = self.builder.load(self.builder.gep(elem_ptr_boxed, [z, ir.Constant(I32, 2)], inbounds=True))
            elem_val = self._boxed_to_value(etag, epay, node)
            self.list_append(tuple_elem, elem_val)
        self.list_append(result, tuple_elem)
        self.builder.store(self.builder.add(ci, ir.Constant(I64, 1)), idx_a)
        self.builder.branch(cond_bb)
        self.builder.position_at_end(end_bb)
        return result

    def _handle_builtin_enumerate(self, node: ast.Call) -> Value:
        if len(node.args) < 1:
            raise CompileError("enumerate() wymaga co najmniej 1 argumentu", node)
        iterable = self.visit(node.args[0])
        start = 0
        if len(node.args) > 1:
            sv = self.visit(node.args[1])
            start = self._get_const_int(sv) or 0
        # Obsługa keyword argumentu start=
        for kw in node.keywords:
            if kw.arg == "start":
                sv = self.visit(kw.value)
                start = self._get_const_int(sv) or 0
        z = ir.Constant(I32, 0)
        func = self.current_func

        if iterable.is_list:
            sp, _, dp = self._list_ptrs(iterable.llvm)
            list_size = self.builder.load(sp, "enum_sz")
            list_data = self.builder.load(dp, "enum_dt")
        elif iterable.is_object:
            tag, pay = self._read_slot(iterable.llvm)
            pay_ptr_generic = self.builder.inttoptr(pay, LIST_PTR)
            list_size = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 1)], inbounds=True))
            list_data = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 3)], inbounds=True))
        else:
            raise CompileError("enumerate() wymaga argumentu typu list lub object", node)

        result = self.create_list([])
        idx_a = self.builder.alloca(I64, name="enum_idx")
        self.builder.store(ir.Constant(I64, 0), idx_a)
        cond_bb = func.append_basic_block("enum.cond")
        body_bb = func.append_basic_block("enum.body")
        end_bb = func.append_basic_block("enum.end")
        self.builder.branch(cond_bb)
        self.builder.position_at_end(cond_bb)
        ci = self.builder.load(idx_a)
        self.builder.cbranch(self.builder.icmp_signed("<", ci, list_size), body_bb, end_bb)
        self.builder.position_at_end(body_bb)
        pair = self.create_tuple([])
        counter = self.builder.add(ir.Constant(I64, start), ci)
        self.list_append(pair, Value(counter, PyType.INT))
        elem_ptr = self.builder.gep(list_data, [ci], inbounds=True)
        elem_ptr_boxed = self.builder.bitcast(elem_ptr, BOXED_PTR)
        etag = self.builder.load(self.builder.gep(elem_ptr_boxed, [z, ir.Constant(I32, 1)], inbounds=True))
        epay = self.builder.load(self.builder.gep(elem_ptr_boxed, [z, ir.Constant(I32, 2)], inbounds=True))
        elem_val = self._boxed_to_value(etag, epay, node)
        self.list_append(pair, elem_val)
        self.list_append(result, pair)
        self.builder.store(self.builder.add(ci, ir.Constant(I64, 1)), idx_a)
        self.builder.branch(cond_bb)
        self.builder.position_at_end(end_bb)
        return result

    def _handle_builtin_map(self, node: ast.Call) -> Value:
        if len(node.args) < 2:
            raise CompileError("map() wymaga co najmniej 2 argumentow", node)
        func_arg = node.args[0]
        iterable = self.visit(node.args[1])
        z = ir.Constant(I32, 0)
        cur_func = self.current_func

        if iterable.is_list:
            sp, _, dp = self._list_ptrs(iterable.llvm)
            list_size = self.builder.load(sp, "map_sz")
            list_data = self.builder.load(dp, "map_dt")
        elif iterable.is_object:
            tag, pay = self._read_slot(iterable.llvm)
            pay_ptr_generic = self.builder.inttoptr(pay, LIST_PTR)
            list_size = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 1)], inbounds=True))
            list_data = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 3)], inbounds=True))
        else:
            raise CompileError("map() wymaga iterowalnego argumentu", node)
        result = self.create_list([])
        idx_a = self.builder.alloca(I64, name="map_idx")
        self.builder.store(ir.Constant(I64, 0), idx_a)
        cond_bb = cur_func.append_basic_block("map.cond")
        body_bb = cur_func.append_basic_block("map.body")
        end_bb = cur_func.append_basic_block("map.end")
        self.builder.branch(cond_bb)
        self.builder.position_at_end(cond_bb)
        ci = self.builder.load(idx_a)
        self.builder.cbranch(self.builder.icmp_signed("<", ci, list_size), body_bb, end_bb)
        self.builder.position_at_end(body_bb)
        elem_ptr = self.builder.gep(list_data, [ci], inbounds=True)
        elem_ptr_boxed = self.builder.bitcast(elem_ptr, BOXED_PTR)
        etag = self.builder.load(self.builder.gep(elem_ptr_boxed, [z, ir.Constant(I32, 1)], inbounds=True))
        epay = self.builder.load(self.builder.gep(elem_ptr_boxed, [z, ir.Constant(I32, 2)], inbounds=True))
        elem_val = self._boxed_to_value(etag, epay, node)
        if isinstance(func_arg, ast.Lambda):
            mapped_val = self._apply_lambda(func_arg, [elem_val], node)
        elif isinstance(func_arg, ast.Name):
            fn_name = func_arg.id
            llvm_name = f"py_{fn_name}"
            if llvm_name in self.functions:
                fn = self.functions[llvm_name]
                boxed = self._box(elem_val)
                ret = self.builder.call(fn, [boxed])
                mapped_val = Value(ret, PyType.OBJECT)
            elif fn_name == "str":
                mapped_val = self.val_to_str(elem_val)
            else:
                raise CompileError(f"map(): nieznana funkcja '{fn_name}'", node)
        else:
            raise CompileError("map(): pierwszy argument musi byc funkcja lub lambda", node)
        self.list_append(result, mapped_val)
        self.builder.store(self.builder.add(ci, ir.Constant(I64, 1)), idx_a)
        self.builder.branch(cond_bb)
        self.builder.position_at_end(end_bb)
        return result

    def _handle_builtin_filter(self, node: ast.Call) -> Value:
        if len(node.args) < 2:
            raise CompileError("filter() wymaga co najmniej 2 argumentow", node)
        func_arg = node.args[0]
        iterable = self.visit(node.args[1])
        z = ir.Constant(I32, 0)
        cur_func = self.current_func

        if iterable.is_list:
            sp, _, dp = self._list_ptrs(iterable.llvm)
            list_size = self.builder.load(sp, "filt_sz")
            list_data = self.builder.load(dp, "filt_dt")
        elif iterable.is_object:
            tag, pay = self._read_slot(iterable.llvm)
            pay_ptr_generic = self.builder.inttoptr(pay, LIST_PTR)
            list_size = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 1)], inbounds=True))
            list_data = self.builder.load(self.builder.gep(pay_ptr_generic, [z, ir.Constant(I32, 3)], inbounds=True))
        else:
            raise CompileError("filter() wymaga iterowalnego argumentu", node)
        result = self.create_list([])
        idx_a = self.builder.alloca(I64, name="filt_idx")
        self.builder.store(ir.Constant(I64, 0), idx_a)
        cond_bb = cur_func.append_basic_block("filt.cond")
        body_bb = cur_func.append_basic_block("filt.body")
        keep_bb = cur_func.append_basic_block("filt.keep")
        skip_bb = cur_func.append_basic_block("filt.skip")
        end_bb = cur_func.append_basic_block("filt.end")
        self.builder.branch(cond_bb)
        self.builder.position_at_end(cond_bb)
        ci = self.builder.load(idx_a)
        self.builder.cbranch(self.builder.icmp_signed("<", ci, list_size), body_bb, end_bb)
        self.builder.position_at_end(body_bb)
        elem_ptr = self.builder.gep(list_data, [ci], inbounds=True)
        elem_ptr_boxed = self.builder.bitcast(elem_ptr, BOXED_PTR)
        etag = self.builder.load(self.builder.gep(elem_ptr_boxed, [z, ir.Constant(I32, 1)], inbounds=True))
        epay = self.builder.load(self.builder.gep(elem_ptr_boxed, [z, ir.Constant(I32, 2)], inbounds=True))
        elem_val = self._boxed_to_value(etag, epay, node)
        if isinstance(func_arg, ast.Lambda):
            check_val = self._apply_lambda(func_arg, [elem_val], node)
        elif isinstance(func_arg, ast.Name):
            fn_name = func_arg.id
            llvm_name = f"py_{fn_name}"
            if llvm_name in self.functions:
                fn = self.functions[llvm_name]
                boxed = self._box(elem_val)
                ret = self.builder.call(fn, [boxed])
                check_val = Value(ret, PyType.OBJECT)
            else:
                raise CompileError(f"filter(): nieznana funkcja '{fn_name}'", node)
        else:
            raise CompileError("filter(): pierwszy argument musi byc funkcja lub lambda", node)
        is_truthy = self._eval_truthiness(check_val)
        self.builder.cbranch(is_truthy.llvm, keep_bb, skip_bb)
        self.builder.position_at_end(keep_bb)
        self.list_append(result, elem_val)
        self.builder.branch(skip_bb)
        self.builder.position_at_end(skip_bb)
        self.builder.store(self.builder.add(ci, ir.Constant(I64, 1)), idx_a)
        self.builder.branch(cond_bb)
        self.builder.position_at_end(end_bb)
        return result

    def _apply_lambda(self, lambda_node: ast.Lambda, args: list, ctx_node: ast.AST) -> Value:
        params = lambda_node.args.args
        if len(params) != len(args):
            raise CompileError(f"lambda oczekuje {len(params)} arg, podano {len(args)}", ctx_node)
        old_sym = self.sym
        self.sym = SymbolTable(parent=old_sym)
        for param, arg_val in zip(params, args):
            alloca = self.builder.alloca(BOXED_PTR, name=f"lam_{param.arg}")
            boxed = self._box(arg_val)
            self.builder.store(boxed, alloca)
            self.sym.define(param.arg, VarInfo(alloca, BOXED_PTR, PyType.OBJECT))
        result = self.visit(lambda_node.body)
        self.sym = old_sym
        return result

    def builtin_len(self, obj: Value) -> Value:
        z = ir.Constant(I32, 0)
        if obj.is_str:
            sz = self.builder.load(
                self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
            )
            return Value(sz, PyType.INT)
        if obj.is_list:
            return self.list_len(obj)
        if obj.is_dict:
            return self.dict_len(obj)
        if obj.is_object:
            tag, pay = self._read_slot(obj.llvm)
            lst_bb = self.current_func.append_basic_block("blen.lst")
            dct_bb = self.current_func.append_basic_block("blen.dct")
            err_bb = self.current_func.append_basic_block("blen.err")
            end_bb = self.current_func.append_basic_block("blen.end")
            res = self.builder.alloca(I64, name="blen_res")
            str_bb = self.current_func.append_basic_block("blen.str")

            # Dodaj switch PRZED pozycjonowaniem w str_bb
            sw = self.builder.switch(tag, err_bb)
            sw.add_case(ir.Constant(I64, Tag.LIST), lst_bb)
            sw.add_case(ir.Constant(I64, Tag.DICT), dct_bb)
            sw.add_case(ir.Constant(I64, Tag.STR), str_bb)

            # Teraz pozycjonuj i wypełniaj bloki
            self.builder.position_at_end(str_bb)
            str_ptr = self.builder.inttoptr(pay, STR_PTR)
            sz = self.builder.load(
                self.builder.gep(str_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
            )
            self.builder.store(sz, res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(lst_bb)
            lptr = self.builder.inttoptr(pay, LIST_PTR)
            r = self.list_len(Value(lptr, PyType.LIST))
            self.builder.store(r.llvm, res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(dct_bb)
            dptr = self.builder.inttoptr(pay, DICT_PTR)
            r = self.dict_len(Value(dptr, PyType.DICT))
            self.builder.store(r.llvm, res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(err_bb)
            self.builder.store(ir.Constant(I64, 0), res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(end_bb)
            return Value(self.builder.load(res), PyType.INT)
        raise CompileError(f"len() nie obsługuje {obj.pytype.name}")

    # ──────────────────────────────────────────────────────────────
    #  DYNAMIC BINOP  –  arytmetyka na OBJECT w runtime
    # ──────────────────────────────────────────────────────────────

