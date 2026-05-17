"""String creation, manipulation, and formatting operations."""

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


class StringsMixin:
    """String creation, manipulation, and formatting operations."""

    def create_string(self, py_str: str) -> Value:
        """String Interning - tworzy py_str bezpośrednio w pamięci GlobalVariable"""
        if py_str in self._str_cache_objs:
            return Value(self._str_cache_objs[py_str], PyType.STR)

        enc = (py_str + "\0").encode("utf-8")
        sz = len(enc) - 1

        # Bufor tekstowy
        arr_ty = ir.ArrayType(I8, sz + 1)
        gv_data = ir.GlobalVariable(
            self.module, arr_ty, name=f".sdata.{len(self._str_cache_objs)}"
        )
        gv_data.global_constant = True
        gv_data.initializer = ir.Constant(arr_ty, bytearray(enc))
        gv_data.linkage = "private"

        # Obiekt StringObject z GC_HEADER: {GC_HEADER, len, cap, data}
        gv_obj = ir.GlobalVariable(
            self.module, STR_TY, name=f".sobj.{len(self._str_cache_objs)}"
        )
        gv_obj.global_constant = False
        null_i8p = ir.Constant(I8P, None)
        gc_header = ir.Constant(
            GC_HEADER_TY,
            [
                ir.Constant(I64, 1),  # refcnt = 1
                ir.Constant(I32, -1),  # color = STATIC (niezniszczalny)
                ir.Constant(I64, 0),  # temp_refcnt = 0
                null_i8p,  # gc_next = null
            ],
        )
        gv_obj.initializer = ir.Constant(
            STR_TY,
            [
                gc_header,
                ir.Constant(I64, sz),
                ir.Constant(I64, sz + 1),
                gv_data.bitcast(I8P),
            ],
        )
        gv_obj.linkage = "private"

        self._str_cache_objs[py_str] = gv_obj
        return Value(gv_obj, PyType.STR)

    def create_instance(self, class_ptr: ir.Value, attrs_dict: Value) -> Value:
        """Tworzy obiekt instancji: {GC_HEADER, CLASS_PTR, DICT_PTR}."""
        z = ir.Constant(I32, 0)
        raw = self.builder.call(self._malloc, [ir.Constant(I64, 56)], "inst_raw")
        inst_ptr = self.builder.bitcast(raw, INSTANCE_PTR)
        gc_hdr = self.builder.gep(inst_ptr, [z, z], inbounds=True)
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(gc_hdr, [z, ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, 0), self.builder.gep(gc_hdr, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(gc_hdr, [z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(ir.Constant(I8P, None), self.builder.gep(gc_hdr, [z, ir.Constant(I32, 3)], inbounds=True))
        self.builder.store(class_ptr, self.builder.gep(inst_ptr, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(attrs_dict.llvm, self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True))
        return Value(inst_ptr, PyType.INSTANCE)

    def _create_str_object(
        self, buf: ir.Value, length: ir.Value, capacity: ir.Value
    ) -> Value:

        """Tworzy obiekt String z runtime bufora (malloc'd)."""
        z = ir.Constant(I32, 0)
        str_obj = self.builder.bitcast(buf, STR_PTR)

        # GC_HEADER: { refcnt, color, temp_refcnt, gc_next }
        gc_hdr = self.builder.gep(str_obj, [z, z], inbounds=True)
        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(gc_hdr, [z, z], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(gc_hdr, [z, ir.Constant(I32, 1)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(gc_hdr, [z, ir.Constant(I32, 2)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I8P, None),
            self.builder.gep(gc_hdr, [z, ir.Constant(I32, 3)], inbounds=True),
        )

        # STR_TY: { GC_HEADER, len, cap, data }
        self.builder.store(
            length, self.builder.gep(str_obj, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        self.builder.store(
            capacity, self.builder.gep(str_obj, [z, ir.Constant(I32, 2)], inbounds=True)
        )

        data_ptr = self.builder.call(self._malloc, [capacity])
        self.builder.store(
            data_ptr, self.builder.gep(str_obj, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        return Value(str_obj, PyType.STR)

    def _get_or_create_concat_fn(self) -> ir.Function:
        if "__py2llvm_concat" in self.functions:
            return self.functions["__py2llvm_concat"]

        fty = ir.FunctionType(STR_PTR, [STR_PTR, STR_PTR])
        fn = ir.Function(self.module, fty, "__py2llvm_concat")
        self.functions["__py2llvm_concat"] = fn

        old_b, old_f = self.builder, self.current_func
        self.current_func = fn
        self.builder = ir.IRBuilder(fn.append_basic_block("entry"))

        s1, s2 = fn.args[0], fn.args[1]
        z = ir.Constant(I32, 0)

        sz1 = self.builder.load(
            self.builder.gep(s1, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        sz2 = self.builder.load(
            self.builder.gep(s2, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        new_sz = self.builder.add(sz1, sz2)
        new_cap = self.builder.add(new_sz, ir.Constant(I64, 1))

        d1 = self.builder.load(
            self.builder.gep(s1, [z, ir.Constant(I32, 3)], inbounds=True)
        )
        d2 = self.builder.load(
            self.builder.gep(s2, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        raw_str = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)])
        str_obj = self.builder.bitcast(raw_str, STR_PTR)
        data_ptr = self.builder.call(self._malloc, [new_cap])

        self.builder.call(self._strcpy, [data_ptr, d1])
        self.builder.call(self._strcat, [data_ptr, d2])

        null_i8p = ir.Constant(I8P, None)
        gc_hdr = self.builder.gep(str_obj, [z, z], inbounds=True)
        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(gc_hdr, [z, z], inbounds=True),
        )  # refcnt=1
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(gc_hdr, [z, ir.Constant(I32, 1)], inbounds=True),
        )  # color=Black
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(gc_hdr, [z, ir.Constant(I32, 2)], inbounds=True),
        )  # temp_refcnt=0
        self.builder.store(
            null_i8p, self.builder.gep(gc_hdr, [z, ir.Constant(I32, 3)], inbounds=True)
        )  # gc_next=null
        self.builder.store(
            new_sz, self.builder.gep(str_obj, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        self.builder.store(
            new_cap, self.builder.gep(str_obj, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        self.builder.store(
            data_ptr, self.builder.gep(str_obj, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        self.builder.ret(str_obj)
        self.current_func, self.builder = old_f, old_b
        return fn

    def concat_strings(self, s1: Value, s2: Value) -> Value:
        """Kompiluje instrukcję 'a' + 'b'."""
        z = ir.Constant(I32, 0)

        # Odczyt rozmiarów (nowy layout: GC_HEADER=idx0, len=idx1, cap=idx2, data=idx3)
        sz1 = self.builder.load(self.builder.gep(s1.llvm, [z, ir.Constant(I32, 1)]))
        sz2 = self.builder.load(self.builder.gep(s2.llvm, [z, ir.Constant(I32, 1)]))
        new_sz = self.builder.add(sz1, sz2)
        new_cap = self.builder.add(new_sz, ir.Constant(I64, 1))

        # Odczyt wskaźników
        d1 = self.builder.load(self.builder.gep(s1.llvm, [z, ir.Constant(I32, 3)]))
        d2 = self.builder.load(self.builder.gep(s2.llvm, [z, ir.Constant(I32, 3)]))

        # Tworzenie nowego obiektu z GC_HEADER
        raw_str = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)])
        str_obj = self.builder.bitcast(raw_str, STR_PTR)
        data_ptr = self.builder.call(self._malloc, [new_cap])

        self.builder.call(self._strcpy, [data_ptr, d1])
        self.builder.call(self._strcat, [data_ptr, d2])

        null_i8p = ir.Constant(I8P, None)
        gc_hdr = self.builder.gep(str_obj, [z, z], inbounds=True)
        # Init GC_HEADER: refcnt=1, color=Black, temp_refcnt=0, gc_next=null
        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(gc_hdr, [z, ir.Constant(I32, 0)], inbounds=True),
        )  # refcnt=1
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(gc_hdr, [z, ir.Constant(I32, 1)], inbounds=True),
        )  # color=Black
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(gc_hdr, [z, ir.Constant(I32, 2)], inbounds=True),
        )  # temp_refcnt=0
        self.builder.store(
            null_i8p, self.builder.gep(gc_hdr, [z, ir.Constant(I32, 3)], inbounds=True)
        )  # gc_next=null

        self.builder.store(
            new_sz, self.builder.gep(str_obj, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        self.builder.store(
            new_cap, self.builder.gep(str_obj, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        self.builder.store(
            data_ptr, self.builder.gep(str_obj, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        return Value(str_obj, PyType.STR)

    def _string_format(self, fmt_val: Value, args_val: Value, node=None) -> Value:
        """Obsługa formatowania: 'format %s' % (args) lub 'format %s' % arg.
        
        Parsuje łańcuch formatujący w compile-time i generuje kod LLVM
        konwertujący argumenty na stringi i sklejający je z literałami.
        """
        import re as _re
        
        # Pobierz łańcuch formatujący z AST (musi być stałą)
        fmt_str = None
        if node and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            fmt_str = node.left.value
        if fmt_str is None:
            # Fallback: połącz format z wartością przez val_to_str
            return self.concat_strings(fmt_val, self.val_to_str(args_val))
        
        # Parsuj specyfikatory formatu: %s, %d, %i, %f, %.Nf, %r, %%
        # Podziel na: [(literal, spec), ...] gdzie spec może być None
        parts = []
        last_end = 0
        for m in _re.finditer(r'(%%|%s|%d|%i|%f|%\.\d+f|%r)', fmt_str):
            # Tekst przed specyfikatorem
            if m.start() > last_end:
                parts.append(('literal', fmt_str[last_end:m.start()]))
            spec = m.group(1)
            if spec == '%%':
                parts.append(('literal', '%'))
            else:
                parts.append(('spec', spec))
            last_end = m.end()
        # Tekst po ostatnim specyfikatorze
        if last_end < len(fmt_str):
            parts.append(('literal', fmt_str[last_end:]))
        
        # Zbierz argumenty: jeśli args_val to krotka, rozpakuj elementy
        fmt_args = []
        if node and isinstance(node.right, ast.Tuple):
            for elt in node.right.elts:
                fmt_args.append(self.visit(elt))
        else:
            fmt_args.append(args_val)
        
        # Generuj wynik: start z pustym stringiem, potem sklejaj
        result = self.create_string("")
        arg_idx = 0
        for kind, value in parts:
            if kind == 'literal':
                part_str = self.create_string(value)
                result = self.concat_strings(result, part_str)
            elif kind == 'spec':
                if arg_idx >= len(fmt_args):
                    # Brak argumentu – wstaw pusty string
                    part_str = self.create_string("")
                else:
                    arg = fmt_args[arg_idx]
                    arg_idx += 1
                    if value in ('%s', '%r'):
                        part_str = self.val_to_str(arg)
                    elif value in ('%d', '%i'):
                        part_str = self.val_to_str(self._to_int(arg))
                    elif value == '%f':
                        part_str = self.val_to_str(self._to_float(arg))
                    elif value.startswith('%.') and value.endswith('f'):
                        # %.Nf – formatowanie zmiennoprzecinkowe z precyzją
                        precision = int(value[2:-1])
                        fval = self._to_float(arg)
                        part_str = self._format_float(fval, precision)
                    else:
                        part_str = self.val_to_str(arg)
                result = self.concat_strings(result, part_str)
        return result

    def _format_float(self, fval: Value, precision: int) -> Value:
        """Formatuje liczbę zmiennoprzecinkową z podaną precyzją (np. %.2f)."""
        # Użyj snprintf do formatowania
        fmt_literal = f"%.{precision}f"
        buf_size = 64
        buf = self.builder.call(self._malloc, [ir.Constant(I64, buf_size)], "fmt_buf")
        fmt_ptr = self._str_ptr(fmt_literal)
        self.builder.call(self._snprintf, [buf, ir.Constant(I64, buf_size), fmt_ptr, fval.llvm])
        # Stwórz obiekt string z bufora
        raw_str = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)])
        str_obj = self.builder.bitcast(raw_str, STR_PTR)
        z = ir.Constant(I32, 0)
        null_i8p = ir.Constant(I8P, None)
        gc_hdr = self.builder.gep(str_obj, [z, z], inbounds=True)
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(gc_hdr, [z, ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, 0), self.builder.gep(gc_hdr, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(gc_hdr, [z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(null_i8p, self.builder.gep(gc_hdr, [z, ir.Constant(I32, 3)], inbounds=True))
        # Oblicz długość sformatowanego stringa
        strlen_fn = self.functions.get("strlen") or self._declare_strlen()
        str_len = self.builder.call(strlen_fn, [buf])
        self.builder.store(str_len, self.builder.gep(str_obj, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(ir.Constant(I64, buf_size), self.builder.gep(str_obj, [z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(buf, self.builder.gep(str_obj, [z, ir.Constant(I32, 3)], inbounds=True))
        return Value(str_obj, PyType.STR)

    def _declare_strlen(self):
        """Deklaruje funkcję strlen jeśli nie istnieje."""
        if "strlen" not in self.functions:
            fty = ir.FunctionType(I64, [I8P])
            fn = ir.Function(self.module, fty, name="strlen")
            self.functions["strlen"] = fn
        return self.functions["strlen"]

    # ──────────────────────────────────────────────────────────────
    #  Iterator / Generator runtime
    # ──────────────────────────────────────────────────────────────

    def _str_ptrs(self, s: ir.Value):
        """Zwraca (size_ptr, cap_ptr, data_ptr_ptr). GC_HEADER is at index 0."""
        z = ir.Constant(I32, 0)
        sp = self.builder.gep(s, [z, ir.Constant(I32, 1)], inbounds=True)
        cp = self.builder.gep(s, [z, ir.Constant(I32, 2)], inbounds=True)
        dp = self.builder.gep(s, [z, ir.Constant(I32, 3)], inbounds=True)
        return sp, cp, dp

    def string_getitem(self, str_val: Value, idx_val: Value) -> Value:
        """s[idx] -> BOXED_PTR."""
        z = ir.Constant(I32, 0)
        null_i8p = ir.Constant(I8P, None)

        s_ptr = str_val.llvm
        sp, cp, dp = self._str_ptrs(s_ptr)
        size = self.builder.load(sp, "sg_sz")

        raw_idx = self._to_int(idx_val).llvm
        is_neg = self.builder.icmp_signed("<", raw_idx, ir.Constant(I64, 0))
        adj_idx = self.builder.add(raw_idx, size)
        idx = self.builder.select(is_neg, adj_idx, raw_idx)

        data = self.builder.load(dp, "sd")
        char_ptr = self.builder.gep(data, [idx], inbounds=True)
        char_val = self.builder.load(char_ptr, "char_val")

        # Utwórz nowy string o długości 1
        raw_s = self.builder.call(
            self._malloc, [ir.Constant(I64, SZ_STR)], "sgetitem.raw"
        )
        new_s = self.builder.bitcast(raw_s, STR_PTR, "new_s")
        raw_d = self.builder.call(self._malloc, [ir.Constant(I64, 2)], "sgetitem.data")
        new_d = self.builder.bitcast(raw_d, I8P, "new_d")

        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(new_s, [z, z, ir.Constant(I32, 0)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(new_s, [z, z, ir.Constant(I32, 1)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(new_s, [z, z, ir.Constant(I32, 2)], inbounds=True),
        )
        self.builder.store(
            null_i8p,
            self.builder.gep(new_s, [z, z, ir.Constant(I32, 3)], inbounds=True),
        )

        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(new_s, [z, ir.Constant(I32, 1)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I64, 2),
            self.builder.gep(new_s, [z, ir.Constant(I32, 2)], inbounds=True),
        )
        self.builder.store(
            new_d, self.builder.gep(new_s, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        self.builder.store(char_val, new_d)
        self.builder.store(
            ir.Constant(I8, 0),
            self.builder.gep(new_d, [ir.Constant(I32, 1)], inbounds=True),
        )

        # Now Box it!
        boxed_s = self._box(Value(new_s, PyType.STR))
        # incref the new string object (it's a heap object referenced by the box)
        self.builder.call(
            self.functions["__py2llvm_incref"], [self.builder.bitcast(new_s, I8P)]
        )
        return Value(boxed_s, PyType.OBJECT)

    def _str_ptr(self, s: str) -> ir.Value:
        """Zwraca i8* do globalnej stałej (string + NUL)."""
        if s not in self._str_cache:
            enc = (s + "\0").encode("utf-8")
            arr_ty = ir.ArrayType(I8, len(enc))
            gv = ir.GlobalVariable(
                self.module, arr_ty, name=f".str{len(self._str_cache)}"
            )
            gv.global_constant = True
            gv.initializer = ir.Constant(arr_ty, bytearray(enc))
            gv.linkage = "private"
            self._str_cache[s] = gv
        gv = self._str_cache[s]
        z = ir.Constant(I32, 0)
        return self.builder.gep(gv, [z, z], inbounds=True)

    # ──────────────────────────────────────────────────────────────
    #  Pomocniki typów (unboxed konwersje)
    # ──────────────────────────────────────────────────────────────

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Value:
        """Kompilator Pythonowych f-stringów!"""
        res = self.create_string("")
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                str_val = self.create_string(v.value)
            elif isinstance(v, ast.FormattedValue):
                expr_val = self.visit(v.value)
                str_val = self.val_to_str(expr_val)
            else:
                # Ochrona dla nieoczekiwanych części
                str_val = self.val_to_str(self.visit(v))

            res = self.concat_strings(res, str_val)
        return res

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Value:
        """Kompiluje to co jest w nawiasach {x} w f-stringu"""
        expr_val = self.visit(node.value)
        return self.val_to_str(expr_val)

    def _make_str_object(self, length, capacity, data_ptr) -> Value:
        """Helper: tworzy obiekt STR_TY z podanego bufora danych."""
        z = ir.Constant(I32, 0)
        raw_str = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)])
        new_s = self.builder.bitcast(raw_str, STR_PTR)
        # GC header
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(new_s, [z, z, ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, 0), self.builder.gep(new_s, [z, z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(new_s, [z, z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(ir.Constant(I8P, None), self.builder.gep(new_s, [z, z, ir.Constant(I32, 3)], inbounds=True))
        # Fields
        self.builder.store(length, self.builder.gep(new_s, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(capacity, self.builder.gep(new_s, [z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(data_ptr, self.builder.gep(new_s, [z, ir.Constant(I32, 3)], inbounds=True))
        return Value(new_s, PyType.STR)

