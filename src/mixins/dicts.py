"""Dictionary (hash table) runtime operations."""

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


class DictsMixin:
    """Dictionary (hash table) runtime operations."""

    def _entry_fld(self, ents: ir.Value, idx: ir.Value, fld: int) -> ir.Value:
        e = self.builder.gep(ents, [idx], inbounds=True)
        return self.builder.gep(
            e, [ir.Constant(I32, 0), ir.Constant(I32, fld)], inbounds=True
        )

        # ──────────────────────────────────────────────────────────────
        #  DICT runtime (O(1) Hash Table)
        # ──────────────────────────────────────────────────────────────

    def _dict_ptrs(self, dct: ir.Value):
        z = ir.Constant(I32, 0)
        sp = self.builder.gep(dct, [z, ir.Constant(I32, 1)], inbounds=True)
        cp = self.builder.gep(dct, [z, ir.Constant(I32, 2)], inbounds=True)
        ep = self.builder.gep(dct, [z, ir.Constant(I32, 3)], inbounds=True)
        return sp, cp, ep

    def _ensure_dict_funcs(self):
        """Generuje krytyczne funkcje Hash Table i Search bezpośrednio w LLVM IR."""
        if "__py2llvm_dict_set_internal" in self.functions:
            return

        # Deklaracje wyprzedzające
        hash_ty = ir.FunctionType(I64, [I64, I64])
        hash_fn = ir.Function(self.module, hash_ty, "__py2llvm_hash")
        eq_ty = ir.FunctionType(I1, [I64, I64, I64, I64])
        eq_fn = ir.Function(self.module, eq_ty, "__py2llvm_key_eq")
        set_ty = ir.FunctionType(VOID, [DICT_PTR, I64, I64, I64, I64])
        set_fn = ir.Function(self.module, set_ty, "__py2llvm_dict_set_internal")
        grow_ty = ir.FunctionType(VOID, [DICT_PTR])
        grow_fn = ir.Function(self.module, grow_ty, "__py2llvm_dict_grow")

        # Implementacja __py2llvm_dict_grow (pusta, aby uniknąć błędów linkera)
        grow_entry = grow_fn.append_basic_block("entry")
        grow_builder = ir.IRBuilder(grow_entry)
        grow_builder.ret_void()

        get_ty = ir.FunctionType(
            VOID, [DICT_PTR, I64, I64, I64.as_pointer(), I64.as_pointer()]
        )
        get_fn = ir.Function(self.module, get_ty, "__py2llvm_dict_get_internal")

        list_cont_ty = ir.FunctionType(I1, [LIST_PTR, I64, I64])
        list_cont_fn = ir.Function(self.module, list_cont_ty, "__py2llvm_list_contains")
        dict_cont_ty = ir.FunctionType(I1, [DICT_PTR, I64, I64])
        dict_cont_fn = ir.Function(self.module, dict_cont_ty, "__py2llvm_dict_contains")

        self.functions.update(
            {
                "__py2llvm_hash": hash_fn,
                "__py2llvm_key_eq": eq_fn,
                "__py2llvm_dict_set_internal": set_fn,
                "__py2llvm_dict_grow": grow_fn,
                "__py2llvm_dict_get_internal": get_fn,
                "__py2llvm_list_contains": list_cont_fn,
                "__py2llvm_dict_contains": dict_cont_fn,
            }
        )

        old_b, old_f = self.builder, self.current_func
        z = ir.Constant(I32, 0)

        # 1. Funkcja: FNV-1a Hash
        self.current_func = hash_fn
        self.builder = ir.IRBuilder(hash_fn.append_basic_block("entry"))
        tag, pay = hash_fn.args
        str_bb = hash_fn.append_basic_block("h.str")
        mix_bb = hash_fn.append_basic_block("h.mix")
        self.builder.cbranch(
            self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.STR)),
            str_bb,
            mix_bb,
        )
        self.builder.position_at_end(str_bb)
        str_obj = self.builder.inttoptr(pay, STR_PTR)
        sz = self.builder.load(
            self.builder.gep(str_obj, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        data = self.builder.load(
            self.builder.gep(str_obj, [z, ir.Constant(I32, 3)], inbounds=True)
        )
        h_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0xCBF29CE484222325), h_a)  # FNV-1a basis
        i_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), i_a)
        l_cond = hash_fn.append_basic_block("h.cond")
        l_body = hash_fn.append_basic_block("h.body")
        l_end = hash_fn.append_basic_block("h.end")
        self.builder.branch(l_cond)
        self.builder.position_at_end(l_cond)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(i_a), sz), l_body, l_end
        )
        self.builder.position_at_end(l_body)
        i_val = self.builder.load(i_a)
        byte = self.builder.load(self.builder.gep(data, [i_val], inbounds=True))
        b_ext = self.builder.zext(byte, I64)
        h_val = self.builder.load(h_a)
        h_val = self.builder.xor(h_val, b_ext)
        h_val = self.builder.mul(h_val, ir.Constant(I64, 0x100000001B3))  # FNV-1a prime
        self.builder.store(h_val, h_a)
        self.builder.store(self.builder.add(i_val, ir.Constant(I64, 1)), i_a)
        self.builder.branch(l_cond)
        self.builder.position_at_end(l_end)
        self.builder.ret(self.builder.load(h_a))
        self.builder.position_at_end(mix_bb)
        # Szybki hash dla Int/Float
        m_val = self.builder.xor(pay, ir.Constant(I64, 0xCBF29CE484222325))
        m_val = self.builder.mul(m_val, ir.Constant(I64, 0x100000001B3))
        self.builder.ret(m_val)

        # 3. Funkcja: Key Equality
        # __py2llvm_key_eq(kt1, kp1, kt2, kp2) -> i1
        # Compares two dictionary keys by tag and payload.
        # For STR tags, uses strcmp; for INT/FLOAT/BOOL, compares payloads directly.
        self.current_func = eq_fn
        self.builder = ir.IRBuilder(eq_fn.append_basic_block("entry"))
        kt1, kp1, kt2, kp2 = eq_fn.args

        # Fast path: if tags differ, keys are not equal
        tags_eq = self.builder.icmp_signed("==", kt1, kt2, "tags_eq")
        tags_neq_bb = eq_fn.append_basic_block("eq.tags_neq")
        tags_eq_bb = eq_fn.append_basic_block("eq.tags_eq")
        self.builder.cbranch(tags_eq, tags_eq_bb, tags_neq_bb)

        # Tags not equal → return false
        self.builder.position_at_end(tags_neq_bb)
        self.builder.ret(ir.Constant(I1, 0))

        # Tags equal → compare payloads
        self.builder.position_at_end(tags_eq_bb)

        # Check if both are STR tags → use strcmp
        is_str_tag = self.builder.icmp_signed("==", kt1, ir.Constant(I64, Tag.STR), "is_str_tag")
        str_cmp_bb = eq_fn.append_basic_block("eq.str_cmp")
        pay_cmp_bb = eq_fn.append_basic_block("eq.pay_cmp")
        self.builder.cbranch(is_str_tag, str_cmp_bb, pay_cmp_bb)

        # String comparison: load data pointers from both STR_PTRs and call strcmp
        self.builder.position_at_end(str_cmp_bb)
        str1_ptr = self.builder.inttoptr(kp1, STR_PTR, "str1_ptr")
        str2_ptr = self.builder.inttoptr(kp2, STR_PTR, "str2_ptr")
        str1_data = self.builder.load(
            self.builder.gep(str1_ptr, [z, ir.Constant(I32, 3)], inbounds=True),
            "str1_data"
        )
        str2_data = self.builder.load(
            self.builder.gep(str2_ptr, [z, ir.Constant(I32, 3)], inbounds=True),
            "str2_data"
        )
        strcmp_decl = self.functions.get("strcmp")
        if strcmp_decl is None:
            strcmp_ty = ir.FunctionType(I32, [I8P, I8P])
            strcmp_decl = ir.Function(self.module, strcmp_ty, name="strcmp")
            self.functions["strcmp"] = strcmp_decl
        cmp_result = self.builder.call(strcmp_decl, [str1_data, str2_data], "strcmp_res")
        str_eq = self.builder.icmp_signed("==", cmp_result, ir.Constant(I32, 0), "str_eq")
        self.builder.ret(str_eq)

        # Payload comparison for INT/FLOAT/BOOL/None etc.
        self.builder.position_at_end(pay_cmp_bb)
        pay_eq = self.builder.icmp_signed("==", kp1, kp2, "pay_eq")
        self.builder.ret(pay_eq)

        # 4. Set
        self.current_func = set_fn
        self.builder = ir.IRBuilder(set_fn.append_basic_block("entry"))
        dct, kt, kp, vt, vp = set_fn.args
        d_sz = self.builder.load(
            self.builder.gep(dct, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        d_cap = self.builder.load(
            self.builder.gep(dct, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        grow_bb = set_fn.append_basic_block("s.grow")
        cont_bb = set_fn.append_basic_block("s.cont")
        self.builder.cbranch(
            self.builder.icmp_unsigned(
                ">=",
                self.builder.mul(d_sz, ir.Constant(I64, 3)),
                self.builder.mul(d_cap, ir.Constant(I64, 2)),
            ),
            grow_bb,
            cont_bb,
        )
        self.builder.position_at_end(grow_bb)
        self.builder.call(grow_fn, [dct])
        self.builder.branch(cont_bb)
        self.builder.position_at_end(cont_bb)
        d_cap_now = self.builder.load(
            self.builder.gep(dct, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        d_ents_now = self.builder.load(
            self.builder.gep(dct, [z, ir.Constant(I32, 3)], inbounds=True)
        )
        h = self.builder.call(hash_fn, [kt, kp])
        mask = self.builder.sub(d_cap_now, ir.Constant(I64, 1))
        idx_a = self.builder.alloca(I64)
        self.builder.store(self.builder.and_(h, mask), idx_a)
        s_loop = set_fn.append_basic_block("s.loop")
        self.builder.branch(s_loop)
        self.builder.position_at_end(s_loop)
        idx = self.builder.load(idx_a)
        ent = self.builder.gep(d_ents_now, [idx], inbounds=True)
        cur_t = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 0)], inbounds=True)
        )
        s_empty = set_fn.append_basic_block("s.empty")
        s_check = set_fn.append_basic_block("s.check")
        self.builder.cbranch(
            self.builder.icmp_signed("==", cur_t, ir.Constant(I64, -1)),
            s_empty,
            s_check,
        )
        self.builder.position_at_end(s_empty)
        self.builder.store(
            kt, self.builder.gep(ent, [z, ir.Constant(I32, 0)], inbounds=True)
        )
        self.builder.store(
            kp, self.builder.gep(ent, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        self.builder.store(
            vt, self.builder.gep(ent, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        self.builder.store(
            vp, self.builder.gep(ent, [z, ir.Constant(I32, 3)], inbounds=True)
        )
        self.builder.store(
            self.builder.add(
                self.builder.load(
                    self.builder.gep(dct, [z, ir.Constant(I32, 1)], inbounds=True)
                ),
                ir.Constant(I64, 1),
            ),
            self.builder.gep(dct, [z, ir.Constant(I32, 1)], inbounds=True),
        )
        self.builder.ret_void()
        self.builder.position_at_end(s_check)
        cur_p = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        is_eq = self.builder.call(eq_fn, [kt, kp, cur_t, cur_p])
        s_update = set_fn.append_basic_block("s.update")
        s_next = set_fn.append_basic_block("s.next")
        self.builder.cbranch(is_eq, s_update, s_next)
        self.builder.position_at_end(s_update)
        self.builder.store(
            vt, self.builder.gep(ent, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        self.builder.store(
            vp, self.builder.gep(ent, [z, ir.Constant(I32, 3)], inbounds=True)
        )
        self.builder.ret_void()
        self.builder.position_at_end(s_next)
        self.builder.store(
            self.builder.and_(self.builder.add(idx, ir.Constant(I64, 1)), mask), idx_a
        )
        self.builder.branch(s_loop)

        # 5. Get
        self.current_func = get_fn
        self.builder = ir.IRBuilder(get_fn.append_basic_block("entry"))
        dct, kt, kp, res_t, res_p = get_fn.args
        d_cap = self.builder.load(
            self.builder.gep(dct, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        d_ents = self.builder.load(
            self.builder.gep(dct, [z, ir.Constant(I32, 3)], inbounds=True)
        )
        h = self.builder.call(hash_fn, [kt, kp])
        mask = self.builder.sub(d_cap, ir.Constant(I64, 1))
        idx_a = self.builder.alloca(I64)
        self.builder.store(self.builder.and_(h, mask), idx_a)
        g_loop = get_fn.append_basic_block("g.loop")
        self.builder.branch(g_loop)
        self.builder.position_at_end(g_loop)
        idx = self.builder.load(idx_a)
        ent = self.builder.gep(d_ents, [idx], inbounds=True)
        cur_t = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 0)], inbounds=True)
        )
        g_empty = get_fn.append_basic_block("g.empty")
        g_check = get_fn.append_basic_block("g.check")
        self.builder.cbranch(
            self.builder.icmp_signed("==", cur_t, ir.Constant(I64, -1)),
            g_empty,
            g_check,
        )
        self.builder.position_at_end(g_empty)
        self.builder.store(ir.Constant(I64, Tag.NONE), res_t)
        self.builder.store(ir.Constant(I64, 0), res_p)
        self.builder.ret_void()
        self.builder.position_at_end(g_check)
        cur_p = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        is_eq = self.builder.call(eq_fn, [kt, kp, cur_t, cur_p])
        g_found = get_fn.append_basic_block("g.found")
        g_next = get_fn.append_basic_block("g.next")
        self.builder.cbranch(is_eq, g_found, g_next)
        self.builder.position_at_end(g_found)
        self.builder.store(
            self.builder.load(
                self.builder.gep(ent, [z, ir.Constant(I32, 2)], inbounds=True)
            ),
            res_t,
        )
        self.builder.store(
            self.builder.load(
                self.builder.gep(ent, [z, ir.Constant(I32, 3)], inbounds=True)
            ),
            res_p,
        )
        self.builder.ret_void()
        self.builder.position_at_end(g_next)
        # Wrap around: idx = (idx + 1) & mask
        self.builder.store(
            self.builder.and_(self.builder.add(idx, ir.Constant(I64, 1)), mask), idx_a
        )
        self.builder.branch(g_loop)

        # 6. List Contains
        self.current_func = list_cont_fn
        self.builder = ir.IRBuilder(list_cont_fn.append_basic_block("entry"))
        l_ptr, k_tag, k_pay = list_cont_fn.args
        sp, cp, dp = self._list_ptrs(l_ptr)
        size = self.builder.load(sp)
        data = self.builder.load(dp)
        idx_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), idx_a)
        l_cond = list_cont_fn.append_basic_block("lc.cond")
        l_body = list_cont_fn.append_basic_block("lc.body")
        l_end = list_cont_fn.append_basic_block("lc.end")
        self.builder.branch(l_cond)
        self.builder.position_at_end(l_cond)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(idx_a), size), l_body, l_end
        )
        self.builder.position_at_end(l_body)
        slot = self.builder.gep(data, [self.builder.load(idx_a)], inbounds=True)
        cur_tag = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        cur_pay = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        eq_check = self.builder.and_(
            self.builder.icmp_signed("==", cur_tag, k_tag),
            self.builder.icmp_signed("==", cur_pay, k_pay),
        )
        found_bb = list_cont_fn.append_basic_block("lc.found")
        l_next = list_cont_fn.append_basic_block("lc.next")
        self.builder.cbranch(eq_check, found_bb, l_next)
        self.builder.position_at_end(l_next)
        self.builder.store(
            self.builder.add(self.builder.load(idx_a), ir.Constant(I64, 1)), idx_a
        )
        self.builder.branch(l_cond)
        self.builder.position_at_end(found_bb)
        self.builder.ret(ir.Constant(I1, 1))
        self.builder.position_at_end(l_end)
        self.builder.ret(ir.Constant(I1, 0))

        # 7. Dict Contains
        self.current_func = dict_cont_fn
        self.builder = ir.IRBuilder(dict_cont_fn.append_basic_block("entry"))
        d_ptr, k_tag, k_pay = dict_cont_fn.args
        sp, cp, dp = self._dict_ptrs(d_ptr)
        size = self.builder.load(sp)
        cap = self.builder.load(cp)
        ents = self.builder.load(dp)
        h = self.builder.call(hash_fn, [k_tag, k_pay])
        mask = self.builder.sub(cap, ir.Constant(I64, 1))
        idx_a = self.builder.alloca(I64)
        self.builder.store(self.builder.and_(h, mask), idx_a)
        d_cond = dict_cont_fn.append_basic_block("dc.cond")
        d_body = dict_cont_fn.append_basic_block("dc.body")
        d_end = dict_cont_fn.append_basic_block("dc.end")
        self.builder.branch(d_cond)
        self.builder.position_at_end(d_cond)
        cur_idx = self.builder.load(idx_a)
        ent = self.builder.gep(ents, [cur_idx], inbounds=True)
        ent_tag = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 0)], inbounds=True)
        )
        is_empty = self.builder.icmp_signed("==", ent_tag, ir.Constant(I64, -1))
        self.builder.cbranch(is_empty, d_end, d_body)
        self.builder.position_at_end(d_body)
        # NAPRAWA: Entry struktura to [0]=key_tag, [1]=key_payload, [2]=value_tag, [3]=value_payload
        # eq_fn(kt1, kp1, kt2, kp2) potrzebuje key_tag z field 0 i key_payload z field 1
        # Było błędnie: eq_fn(field1, field2, k_tag, k_pay) → eq_fn(kp, vt, kt, kpay) - ZŁE!
        ent_k_pay = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        is_eq = self.builder.call(eq_fn, [ent_tag, ent_k_pay, k_tag, k_pay])
        found_bb = dict_cont_fn.append_basic_block("dc.found")
        d_next = dict_cont_fn.append_basic_block("dc.next")
        self.builder.cbranch(is_eq, found_bb, d_next)
        self.builder.position_at_end(d_next)
        self.builder.store(
            self.builder.and_(self.builder.add(cur_idx, ir.Constant(I64, 1)), mask),
            idx_a,
        )
        self.builder.branch(d_cond)
        self.builder.position_at_end(found_bb)
        self.builder.ret(ir.Constant(I1, 1))
        self.builder.position_at_end(d_end)
        self.builder.ret(ir.Constant(I1, 0))

        self.current_func, self.builder = old_f, old_b

    def create_dict(self, items: List[Tuple[Value, Value]]) -> Value:
        self._ensure_dict_funcs()
        n = len(items)
        cap = 8
        while cap * 2 <= n * 3:
            cap *= 2

        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_DICT)], "dct.raw")
        dct = self.builder.bitcast(raw, DICT_PTR, "dct")
        rawe = self.builder.call(self._malloc, [ir.Constant(I64, cap * SZ_ENTRY)], "ent.raw")
        ents = self.builder.bitcast(rawe, ENTRY_PTR, "ents")

        z = ir.Constant(I32, 0)
        null_i8p = ir.Constant(I8P, None)
        # GC_HEADER
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(dct, [z, z, ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, 0), self.builder.gep(dct, [z, z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(dct, [z, z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(null_i8p, self.builder.gep(dct, [z, z, ir.Constant(I32, 3)], inbounds=True))
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(dct, [z, ir.Constant(I32, 1)], inbounds=True))  # size
        self.builder.store(ir.Constant(I64, cap), self.builder.gep(dct, [z, ir.Constant(I32, 2)], inbounds=True)) # cap
        self.builder.store(ents, self.builder.gep(dct, [z, ir.Constant(I32, 3)], inbounds=True))                # entries

        # ordered_keys list
        ordered = self.create_list([])
        self.builder.store(ordered.llvm, self.builder.gep(dct, [z, ir.Constant(I32, 4)], inbounds=True))

        # Init entries to -1
        i_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), i_a)
        l_cond = self.current_func.append_basic_block("d.init.cond")
        l_body = self.current_func.append_basic_block("d.init.body")
        l_end = self.current_func.append_basic_block("d.init.end")
        self.builder.branch(l_cond)
        self.builder.position_at_end(l_cond)
        self.builder.cbranch(self.builder.icmp_signed("<", self.builder.load(i_a), ir.Constant(I64, cap)), l_body, l_end)
        self.builder.position_at_end(l_body)
        ent = self.builder.gep(ents, [self.builder.load(i_a)], inbounds=True)
        self.builder.store(ir.Constant(I64, -1), self.builder.gep(ent, [z, ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(self.builder.add(self.builder.load(i_a), ir.Constant(I64, 1)), i_a)
        self.builder.branch(l_cond)
        self.builder.position_at_end(l_end)

        set_fn = self.functions["__py2llvm_dict_set_internal"]
        contains_fn = self.functions["__py2llvm_dict_contains"]

        for k, v in items:
            kt, kp = self._value_to_tag_payload(k)
            # 1. Sprawdź czy klucz jest nowy (dla zachowania kolejności w słowniku)
            exists = self.builder.call(contains_fn, [dct, kt, kp])
            new_key_bb = self.current_func.append_basic_block("dset.newkey")
            add_bb = self.current_func.append_basic_block("dset.add")

            self.builder.cbranch(exists, add_bb, new_key_bb)

            self.builder.position_at_end(new_key_bb)
            ordered_list = self.builder.load(self.builder.gep(dct, [z, ir.Constant(I32, 4)], inbounds=True))
            self.list_append(Value(ordered_list, PyType.LIST), k)
            self.builder.branch(add_bb)

            self.builder.position_at_end(add_bb)
            # 2. Ustaw wartość
            vt, vp = self._value_to_tag_payload(v)
            self.builder.call(set_fn, [dct, kt, kp, vt, vp])

            # 3. Zarządzanie pamięcią (Reference Counting)
            # Tutaj możesz dodać te increfy z "drugiej wersji" jeśli są potrzebne
            for val in [k, v]:
                if val.is_object or val.is_list or val.is_dict or val.is_str:
                    ptr = self.builder.bitcast(val.llvm, I8P)
                    self.builder.call(self.functions["__py2llvm_incref"], [ptr])

        return Value(dct, PyType.DICT)
            # self._ensure_dict_funcs()
            # n = len(items)
            # # Pojemność to zawsze potęga dwójki, by `idx = hash & (cap - 1)` działało
            # cap = 8
            # while cap * 2 <= n * 3:
            #     cap *= 2

            # raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_DICT)], "dct.raw")
            # dct = self.builder.bitcast(raw, DICT_PTR, "dct")
            # rawe = self.builder.call(
            #     self._malloc, [ir.Constant(I64, cap * SZ_ENTRY)], "ent.raw"
            # )
            # ents = self.builder.bitcast(rawe, ENTRY_PTR, "ents")

            # z = ir.Constant(I32, 0)
            # # Init GC_HEADER: refcnt=1, color=Black, temp_refcnt=0, gc_next=null
            # null_i8p = ir.Constant(I8P, None)
            # self.builder.store(
            #     ir.Constant(I64, 1),
            #     self.builder.gep(dct, [z, z, ir.Constant(I32, 0)], inbounds=True),
            # )  # refcnt=1
            # self.builder.store(
            #     ir.Constant(I32, 0),
            #     self.builder.gep(dct, [z, z, ir.Constant(I32, 1)], inbounds=True),
            # )  # color=Black
            # self.builder.store(
            #     ir.Constant(I64, 0),
            #     self.builder.gep(dct, [z, z, ir.Constant(I32, 2)], inbounds=True),
            # )  # temp_refcnt=0
            # self.builder.store(
            #     null_i8p, self.builder.gep(dct, [z, z, ir.Constant(I32, 3)], inbounds=True)
            # )  # gc_next=null
            # # size, cap, entries (indices 1, 2, 3 due to GC_HEADER at 0)
            # self.builder.store(
            #     ir.Constant(I64, 0),
            #     self.builder.gep(dct, [z, ir.Constant(I32, 1)], inbounds=True),
            # )
            # self.builder.store(
            #     ir.Constant(I64, cap),
            #     self.builder.gep(dct, [z, ir.Constant(I32, 2)], inbounds=True),
            # )
            # self.builder.store(
            #     ents, self.builder.gep(dct, [z, ir.Constant(I32, 3)], inbounds=True)
            # )

            # # Initialize ordered_keys list
            # ordered_keys = self.create_list([])
            # self.builder.store(
            #     ordered_keys.llvm, self.builder.gep(dct, [z, ir.Constant(I32, 4)], inbounds=True)
            # )

            # # Inicjalizacja tablicy pustymi wartościami (-1)
            # i_a = self.builder.alloca(I64)
            # self.builder.store(ir.Constant(I64, 0), i_a)
            # l_cond = self.current_func.append_basic_block("d.init.cond")
            # l_body = self.current_func.append_basic_block("d.init.body")
            # l_end = self.current_func.append_basic_block("d.init.end")
            # self.builder.branch(l_cond)
            # self.builder.position_at_end(l_cond)
            # self.builder.cbranch(
            #     self.builder.icmp_signed(
            #         "<", self.builder.load(i_a), ir.Constant(I64, cap)
            #     ),
            #     l_body,
            #     l_end,
            # )
            # self.builder.position_at_end(l_body)
            # ent = self.builder.gep(ents, [self.builder.load(i_a)], inbounds=True)
            # self.builder.store(
            #     ir.Constant(I64, -1),
            #     self.builder.gep(ent, [z, ir.Constant(I32, 0)], inbounds=True),
            # )
            # self.builder.store(
            #     self.builder.add(self.builder.load(i_a), ir.Constant(I64, 1)), i_a
            # )
            # self.builder.branch(l_cond)
            # self.builder.position_at_end(l_end)

            # set_fn = self.functions["__py2llvm_dict_set_internal"]
            # for k, v in items:
            #     kt, kp = self._value_to_tag_payload(k)
            #     vt, vp = self._value_to_tag_payload(v)
            #     self.builder.call(set_fn, [dct, kt, kp, vt, vp])
            #     # incref key and value if they are heap objects
            #     if k.is_object or k.is_list or k.is_dict or k.is_str:
            #         self.builder.call(
            #             self.functions["__py2llvm_incref"],
            #             [self.builder.bitcast(k.llvm, I8P)],
            #         )
            #     if v.is_object or v.is_list or v.is_dict or v.is_str:
            #         self.builder.call(
            #             self.functions["__py2llvm_incref"],
            #             [self.builder.bitcast(v.llvm, I8P)],
            #         )

            # return Value(dct, PyType.DICT)

    def dict_setitem(self, dct_val: Value, key: Value, val: Value):
        self._ensure_dict_funcs()
        set_fn = self.functions["__py2llvm_dict_set_internal"]
        kt, kp = self._value_to_tag_payload(key)
        vt, vp = self._value_to_tag_payload(val)

        # Maintain insertion order: if key is new, append to ordered_keys list
        z = ir.Constant(I32, 0)
        dict_cont_fn = self.functions["__py2llvm_dict_contains"]
        exists = self.builder.call(dict_cont_fn, [dct_val.llvm, kt, kp])

        new_key_bb = self.current_func.append_basic_block("dset.new_key")
        set_bb = self.current_func.append_basic_block("dset.set")
        end_bb = self.current_func.append_basic_block("dset.end")

        self.builder.cbranch(exists, set_bb, new_key_bb)

        self.builder.position_at_end(new_key_bb)
        ordered_list_ptr = self.builder.load(
            self.builder.gep(dct_val.llvm, [z, ir.Constant(I32, 4)], inbounds=True)
        )
        # Wrap key as a Value(PyType.LIST) because list_append expects it
        self.list_append(Value(ordered_list_ptr, PyType.LIST), key)
        self.builder.branch(set_bb)

        self.builder.position_at_end(set_bb)
        self.builder.call(set_fn, [dct_val.llvm, kt, kp, vt, vp])
        self.builder.branch(end_bb)

        self.builder.position_at_end(end_bb)

    def dict_getitem(self, dct_val: Value, key: Value) -> Value:
        self._ensure_dict_funcs()
        get_fn = self.functions["__py2llvm_dict_get_internal"]
        kt, kp = self._value_to_tag_payload(key)

        res_t = self.builder.alloca(I64)
        res_p = self.builder.alloca(I64)
        self.builder.call(get_fn, [dct_val.llvm, kt, kp, res_t, res_p])

        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)])
        bv = self.builder.bitcast(raw, BOXED_PTR)
        z = ir.Constant(I32, 0)
        null_i8p = ir.Constant(I8P, None)
        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 0)], inbounds=True),
        )  # refcnt=1
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 1)], inbounds=True),
        )  # color=Black
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 2)], inbounds=True),
        )  # temp_refcnt=0
        self.builder.store(
            null_i8p, self.builder.gep(bv, [z, z, ir.Constant(I32, 3)], inbounds=True)
        )  # gc_next=null
        self.builder.store(
            self.builder.load(res_t),
            self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True),
        )
        self.builder.store(
            self.builder.load(res_p),
            self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True),
        )
        # NAPRAWA: Incref na wewnętrznym obiekcie (tag+payload).
        # Gdy boxed wrapper jest dealokowany, dealloc dekrementuje refcount
        # wewnętrznego obiektu (dla tagów DICT, LIST, STR). Bez tego increfu,
        # pobranie z dict i przypisanie do zmiennej lokalnej powoduje, że
        # dealokacja wrappera zmniejsza refcount obiektu z _db do 0 i go zwalnia.
        val_pay_i64 = self.builder.load(res_p)
        is_heap = self.builder.or_(
            self.builder.or_(
                self.builder.icmp_signed("==", self.builder.load(res_t), ir.Constant(I64, Tag.LIST)),
                self.builder.icmp_signed("==", self.builder.load(res_t), ir.Constant(I64, Tag.DICT)),
            ),
            self.builder.or_(
                self.builder.icmp_signed("==", self.builder.load(res_t), ir.Constant(I64, Tag.STR)),
                self.builder.icmp_signed("==", self.builder.load(res_t), ir.Constant(I64, Tag.SET)),
            ),
        )
        incref_bb = self.current_func.append_basic_block("dgi.incref")
        noincref_bb = self.current_func.append_basic_block("dgi.noincref")
        self.builder.cbranch(is_heap, incref_bb, noincref_bb)
        self.builder.position_at_end(incref_bb)
        inner_ptr = self.builder.inttoptr(val_pay_i64, I8P)
        self.builder.call(self.functions["__py2llvm_incref"], [inner_ptr])
        self.builder.branch(noincref_bb)
        self.builder.position_at_end(noincref_bb)
        return Value(bv, PyType.OBJECT)

    def dict_len(self, dct_val: Value) -> Value:
        sp, _, _ = self._dict_ptrs(dct_val.llvm)
        return Value(self.builder.load(sp, "dlen"), PyType.INT)

    # ──────────────────────────────────────────────────────────────
    #  CLASS & INSTANCE runtime (simplified)
    # ──────────────────────────────────────────────────────────────

