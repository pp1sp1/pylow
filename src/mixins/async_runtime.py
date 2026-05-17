"""Async runtime: C scheduler declarations and coroutine infrastructure.

Implements true cooperative multitasking using POSIX ucontext for context
switching between coroutines. Each async function compiles to a regular
LLVM function plus a coroutine entry wrapper. The C runtime provides:

  - __async_spawn(fn, arg)      — create a new coroutine task
  - __async_run(fn, arg)        — run event loop until main task completes
  - __async_sleep(seconds)      — non-blocking sleep (yields to scheduler)
  - __async_await_task(task)    — wait for another task to complete
  - __async_task_done(task)     — check if task is done
  - __async_task_result(task)   — get result of completed task
  - __async_set_result(result)  — store result in current task
  - __async_yield()             — cooperative yield
  - __async_current_time()      — get monotonic time in seconds
"""

from __future__ import annotations

import ast
import os
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


# ══════════════════════════════════════════════════════════════════
#  C RUNTIME SOURCE — compiled and linked with the LLVM IR
# ══════════════════════════════════════════════════════════════════

ASYNC_RUNTIME_C = r"""
#define _GNU_SOURCE
#include <ucontext.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

/* ── Task states ────────────────────────────────────────────── */
#define TASK_READY    0
#define TASK_RUNNING  1
#define TASK_SLEEPING 2
#define TASK_WAITING  3
#define TASK_DONE     4

/* ── BoxedValue (must match LLVM IR struct layout) ─────────── */
/* { GC_HEADER: {refcnt:i64, color:i32, pad:i32, temp_refcnt:i64, gc_next:i8*},
     tag:i64, payload:i64 }                                      */
typedef struct {
    long refcnt;
    int  color;
    int  _pad0;
    long temp_refcnt;
    void* gc_next;
    long tag;
    long payload;
} BoxedValue;

/* ── Task structure ─────────────────────────────────────────── */
typedef struct Task {
    int              state;
    ucontext_t       ctx;
    BoxedValue*      result;
    double           wake_time;
    int              task_id;
    struct Task*     waiting_for;
    struct Task*     waiters;        /* linked list of tasks waiting for us */
    struct Task*     next_waiter;    /* next in waiters list */
    struct Task*     queue_next;     /* ready queue / sleeping list */
    void             (*entry_fn)(void*);
    void*            entry_arg;
    void*            stack;
    unsigned         stack_size;
} Task;

/* ── Scheduler state ────────────────────────────────────────── */
static Task*    ready_head       = NULL;
static Task*    ready_tail       = NULL;
static Task*    sleeping_head    = NULL;
static Task*    current_task     = NULL;
static ucontext_t scheduler_ctx;
static int      next_task_id     = 1;

/* ── Queue operations ──────────────────────────────────────── */
static void enqueue_ready(Task* task) {
    task->queue_next = NULL;
    if (ready_tail) {
        ready_tail->queue_next = task;
    } else {
        ready_head = task;
    }
    ready_tail = task;
}

static Task* dequeue_ready(void) {
    if (!ready_head) return NULL;
    Task* task = ready_head;
    ready_head = task->queue_next;
    if (!ready_head) ready_tail = NULL;
    task->queue_next = NULL;
    return task;
}

static void add_sleeping(Task* task) {
    task->queue_next = sleeping_head;
    sleeping_head = task;
}

/* ── Wake up tasks that are waiting for a completed task ────── */
static void wake_waiters(Task* completed) {
    Task* w = completed->waiters;
    completed->waiters = NULL;
    while (w) {
        Task* next_w = w->next_waiter;
        w->waiting_for = NULL;
        w->state = TASK_READY;
        w->next_waiter = NULL;
        enqueue_ready(w);
        w = next_w;
    }
}

/* ── Time utilities ────────────────────────────────────────── */
static double get_monotonic_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── Wake up sleeping tasks whose time has come ────────────── */
static void wake_sleeping_tasks(void) {
    double now = get_monotonic_time();
    Task** pp = &sleeping_head;
    while (*pp) {
        Task* t = *pp;
        if (t->wake_time <= now) {
            *pp = t->queue_next;
            t->queue_next = NULL;
            t->state = TASK_READY;
            t->wake_time = 0.0;
            enqueue_ready(t);
        } else {
            pp = &t->queue_next;
        }
    }
}

/* ── Coroutine entry wrapper ───────────────────────────────── */
static void __pyco_coro_entry(void) {
    Task* task = current_task;
    if (task && task->entry_fn) {
        task->entry_fn(task->entry_arg);
    }
    if (task) {
        task->state = TASK_DONE;
        wake_waiters(task);
    }
    /* Return to scheduler */
    setcontext(&scheduler_ctx);
    /* Should never reach here */
    abort();
}

/* ══════════════════════════════════════════════════════════════
 *  PUBLIC API — called from generated LLVM IR
 * ══════════════════════════════════════════════════════════════ */

/* Spawn a new coroutine task */
Task* __async_spawn(void (*fn)(void*), void* arg) {
    Task* task = (Task*)calloc(1, sizeof(Task));
    if (!task) return NULL;
    task->state     = TASK_READY;
    task->task_id   = next_task_id++;
    task->entry_fn  = fn;
    task->entry_arg = arg;
    task->stack_size = 1 << 16;  /* 64 KB */
    task->stack     = malloc(task->stack_size);
    if (!task->stack) { free(task); return NULL; }

    getcontext(&task->ctx);
    task->ctx.uc_stack.ss_sp   = task->stack;
    task->ctx.uc_stack.ss_size = task->stack_size;
    task->ctx.uc_link          = NULL;  /* we handle return via __pyco_coro_entry */
    makecontext(&task->ctx, __pyco_coro_entry, 0);

    enqueue_ready(task);
    return task;
}

/* Yield to the scheduler (cooperative yield) */
void __async_yield(void) {
    if (!current_task) return;
    current_task->state = TASK_READY;
    swapcontext(&current_task->ctx, &scheduler_ctx);
}

/* Sleep for seconds (non-blocking if in scheduler, blocking otherwise) */
void __async_sleep(double seconds) {
    if (!current_task) {
        /* Not inside scheduler — fall back to blocking nanosleep */
        if (seconds > 0.0) {
            struct timespec req;
            req.tv_sec  = (time_t)seconds;
            req.tv_nsec = (long)((seconds - (double)req.tv_sec) * 1e9);
            nanosleep(&req, NULL);
        }
        return;
    }
    if (seconds <= 0.0) {
        /* sleep(0) = just yield to scheduler */
        current_task->state = TASK_READY;
        swapcontext(&current_task->ctx, &scheduler_ctx);
    } else {
        current_task->state     = TASK_SLEEPING;
        current_task->wake_time = get_monotonic_time() + seconds;
        swapcontext(&current_task->ctx, &scheduler_ctx);
    }
}

/* Wait for a task to complete (non-blocking if already done) */
void __async_await_task(Task* task) {
    if (!current_task) return;
    if (!task || task->state == TASK_DONE) return;

    /* Add current task to the waiters list of the target task */
    current_task->waiting_for = task;
    current_task->next_waiter = task->waiters;
    task->waiters = current_task;
    current_task->state = TASK_WAITING;
    swapcontext(&current_task->ctx, &scheduler_ctx);
}

/* Check if task is done */
int __async_task_done(Task* task) {
    return task && task->state == TASK_DONE;
}

/* Get task result */
BoxedValue* __async_task_result(Task* task) {
    if (!task) return NULL;
    return task->result;
}

/* Set task result (called from coroutine entry wrapper) */
void __async_set_result(BoxedValue* result) {
    if (current_task) {
        current_task->result = result;
    }
}

/* Get current monotonic time in seconds */
double __async_current_time(void) {
    return get_monotonic_time();
}

/* Run the scheduler event loop with a main task */
BoxedValue* __async_run(void (*fn)(void*), void* arg) {
    /* Spawn the main coroutine */
    Task* main_task = __async_spawn(fn, arg);
    if (!main_task) return NULL;

    /* Run the event loop until main task completes */
    while (main_task->state != TASK_DONE) {
        /* Wake up sleeping tasks whose time has come */
        wake_sleeping_tasks();

        /* Get next ready task */
        Task* task = dequeue_ready();
        if (!task) {
            /* No ready tasks — check sleeping list */
            if (sleeping_head) {
                /* Wait briefly for sleeping tasks to wake */
                struct timespec ts = {0, 500000};  /* 0.5 ms */
                nanosleep(&ts, NULL);
                continue;
            }
            /* No tasks at all — deadlock or done */
            break;
        }

        /* Execute the task */
        current_task = task;
        task->state = TASK_RUNNING;
        swapcontext(&scheduler_ctx, &task->ctx);
        current_task = NULL;

        /* Task yielded back to us — re-queue or handle */
        if (task->state == TASK_DONE) {
            /* Task completed — wake waiters (already done in __pyco_coro_entry) */
            /* Don't free: result pointer is still needed */
        } else if (task->state == TASK_SLEEPING) {
            add_sleeping(task);
        } else if (task->state == TASK_WAITING) {
            /* Don't re-queue — will be woken by the task it's waiting for */
        } else {
            /* TASK_READY — re-queue */
            enqueue_ready(task);
        }
    }

    BoxedValue* result = main_task->result;

    /* Clean up: free all remaining tasks */
    /* For simplicity, we don't free — the OS will reclaim on exit */
    /* A production implementation would walk all task lists and free them */

    return result;
}
"""


class AsyncRuntimeMixin:
    """Declares async runtime C functions as LLVM IR externals and provides
    coroutine infrastructure for compiling async/await to native code.

    Architecture
    ~~~~~~~~~~~~
    Each ``async def foo(a, b)`` compiles to:

    1. ``py_foo(a, b) -> BOXED_PTR`` — regular LLVM function (no changes)
    2. ``py_foo_coro_entry(i8* arg) -> void`` — coroutine entry wrapper that:
       - Unpacks arguments from the heap-allocated arg struct
       - Calls ``py_foo(a, b)`` to get the result
       - Calls ``__async_set_result(result)`` to store it in the Task
       - Returns (control goes back to the C scheduler)

    When an async function is called from an async context, instead of a
    direct call we:
    1. Allocate an args struct on the heap
    2. Store the arguments
    3. Call ``__async_spawn(py_foo_coro_entry, args)`` → Task*
    4. Return the Task* as a boxed value with Tag.TASK

    ``await`` then:
    1. If the value is a Task (Tag.TASK), call ``__async_await_task`` and
       ``__async_task_result``
    2. If the value is not a Task (e.g. None from sleep), just return it
    """

    def _declare_async_runtime(self):
        """Declare C runtime functions as LLVM IR externals.

        Only called once during compiler initialisation.
        """
        # Task* __async_spawn(void (*fn)(void*), void* arg)
        fn_ptr_ty = ir.PointerType(ir.FunctionType(VOID, [I8P]))
        if "__async_spawn" not in self.functions:
            fty = ir.FunctionType(I8P, [fn_ptr_ty, I8P])
            fn = ir.Function(self.module, fty, name="__async_spawn")
            self.functions["__async_spawn"] = fn

        # void __async_yield()
        if "__async_yield" not in self.functions:
            fty = ir.FunctionType(VOID, [])
            fn = ir.Function(self.module, fty, name="__async_yield")
            self.functions["__async_yield"] = fn

        # void __async_sleep(double)
        if "__async_sleep" not in self.functions:
            fty = ir.FunctionType(VOID, [F64])
            fn = ir.Function(self.module, fty, name="__async_sleep")
            self.functions["__async_sleep"] = fn

        # void __async_await_task(Task*)
        if "__async_await_task" not in self.functions:
            fty = ir.FunctionType(VOID, [I8P])
            fn = ir.Function(self.module, fty, name="__async_await_task")
            self.functions["__async_await_task"] = fn

        # int __async_task_done(Task*)
        if "__async_task_done" not in self.functions:
            fty = ir.FunctionType(I32, [I8P])
            fn = ir.Function(self.module, fty, name="__async_task_done")
            self.functions["__async_task_done"] = fn

        # BoxedValue* __async_task_result(Task*)
        if "__async_task_result" not in self.functions:
            fty = ir.FunctionType(BOXED_PTR, [I8P])
            fn = ir.Function(self.module, fty, name="__async_task_result")
            self.functions["__async_task_result"] = fn

        # void __async_set_result(BoxedValue*)
        if "__async_set_result" not in self.functions:
            fty = ir.FunctionType(VOID, [BOXED_PTR])
            fn = ir.Function(self.module, fty, name="__async_set_result")
            self.functions["__async_set_result"] = fn

        # double __async_current_time()
        if "__async_current_time" not in self.functions:
            fty = ir.FunctionType(F64, [])
            fn = ir.Function(self.module, fty, name="__async_current_time")
            self.functions["__async_current_time"] = fn

        # BoxedValue* __async_run(void (*fn)(void*), void* arg)
        fn_ptr_ty2 = ir.PointerType(ir.FunctionType(VOID, [I8P]))
        if "__async_run" not in self.functions:
            fty = ir.FunctionType(BOXED_PTR, [fn_ptr_ty2, I8P])
            fn = ir.Function(self.module, fty, name="__async_run")
            self.functions["__async_run"] = fn

    # ── Coroutine entry wrapper generation ─────────────────────

    def _generate_coro_entry(self, func_name: str, n_args: int):
        """Generate a coroutine entry wrapper for an async function.

        Creates ``py_{name}_coro_entry(i8* arg_ptr)`` that:
        1. Bitcasts arg_ptr to a pointer to the args struct
        2. Loads each argument from the struct
        3. Calls ``py_{name}(arg0, arg1, ...)`` to get the result
        4. Calls ``__async_set_result(result)`` to store it
        5. Returns void
        """
        entry_name = f"py_{func_name}_coro_entry"

        # Get the original function
        orig_name = f"py_{func_name}"
        orig_fn = self.functions.get(orig_name)
        if orig_fn is None:
            return

        # Define the args struct type: { BOXED_PTR, BOXED_PTR, ... }
        args_struct_fields = [BOXED_PTR] * n_args
        args_struct_ty = ir.LiteralStructType(args_struct_fields)
        args_struct_ptr = ir.PointerType(args_struct_ty)

        # Get or create the coro entry function
        # NOTE: _pre_declare may have already created this as a declaration
        # (no basic blocks). We need to add the body to it.
        coro_fn = self.functions.get(entry_name)
        if coro_fn is None:
            fty = ir.FunctionType(VOID, [I8P])
            coro_fn = ir.Function(self.module, fty, name=entry_name)
            self.functions[entry_name] = coro_fn

        # Check if already has a body (basic blocks)
        if coro_fn.blocks:
            return  # Already generated

        # Generate the body
        saved_func = self.current_func
        saved_builder = self.builder
        saved_sym = self.sym

        self.current_func = coro_fn
        self.sym = SymbolTable()
        entry_bb = coro_fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry_bb)

        arg_ptr = coro_fn.args[0]

        # Bitcast i8* to args_struct_ptr
        args_ptr = self.builder.bitcast(arg_ptr, args_struct_ptr, "args")

        # Load each argument from the struct
        z = ir.Constant(I32, 0)
        call_args = []
        for i in range(n_args):
            field_ptr = self.builder.gep(
                args_ptr, [z, ir.Constant(I32, i)], inbounds=True,
                name=f"arg{i}_ptr"
            )
            field_val = self.builder.load(field_ptr, name=f"arg{i}")
            # Match the original function's argument type
            if i < len(orig_fn.args):
                if orig_fn.args[i].type == BOXED_PTR:
                    call_args.append(field_val)
                else:
                    # Cast as needed
                    call_args.append(self.builder.bitcast(field_val, orig_fn.args[i].type))
            else:
                call_args.append(field_val)

        # Call the original function
        if len(call_args) == len(orig_fn.args):
            result = self.builder.call(orig_fn, call_args, name="coro_result")
            # Store result via __async_set_result
            if orig_fn.function_type.return_type == BOXED_PTR:
                self.builder.call(
                    self.functions["__async_set_result"],
                    [result]
                )
            elif orig_fn.function_type.return_type == VOID:
                # Void function — set result to None
                none_boxed = self._box(Value(ir.Constant(I64, 0), PyType.NONE))
                self.builder.call(
                    self.functions["__async_set_result"],
                    [none_boxed]
                )
            else:
                # Box the result first
                boxed = self._box(Value(result, self._llvm_type_to_pytype(orig_fn.function_type.return_type)))
                self.builder.call(
                    self.functions["__async_set_result"],
                    [boxed]
                )
        else:
            # Argument count mismatch — set result to None
            none_boxed = self._box(Value(ir.Constant(I64, 0), PyType.NONE))
            self.builder.call(
                self.functions["__async_set_result"],
                [none_boxed]
            )

        self.builder.ret_void()

        # Restore context
        self.current_func = saved_func
        self.builder = saved_builder
        self.sym = saved_sym

    # ── Spawn an async function as a coroutine task ────────────

    def _spawn_async_call(self, func_name: str, call_args_llvm: list) -> Value:
        """Generate code to spawn an async function as a coroutine task.

        1. Allocates an args struct on the heap
        2. Stores each argument
        3. Calls __async_spawn(py_{name}_coro_entry, args_ptr) → Task*
        4. Returns a Value with PyType.TASK (boxed with Tag.TASK)

        Args:
            func_name: The Python function name (without py_ prefix).
            call_args_llvm: List of LLVM IR values (the arguments to pass).

        Returns:
            A Value with PyType.TASK containing the Task* pointer boxed
            with Tag.TASK.
        """
        n_args = len(call_args_llvm)

        # Ensure the coro entry function exists
        self._generate_coro_entry(func_name, n_args)

        entry_name = f"py_{func_name}_coro_entry"

        # Define args struct type
        args_struct_fields = [BOXED_PTR] * n_args
        args_struct_ty = ir.LiteralStructType(args_struct_fields)
        args_struct_ptr_ty = ir.PointerType(args_struct_ty)

        # Calculate struct size (each field is a pointer = 8 bytes)
        sz = ir.Constant(I64, 8 * max(n_args, 1))

        # Allocate args struct on the heap
        raw = self.builder.call(self._malloc, [sz], "args_raw")
        args_ptr = self.builder.bitcast(raw, args_struct_ptr_ty, "args_ptr")

        # Store each argument
        z = ir.Constant(I32, 0)
        for i, arg_val in enumerate(call_args_llvm):
            # Box the argument if it's not already a BOXED_PTR
            if isinstance(arg_val, Value):
                if arg_val.is_object:
                    boxed_arg = arg_val.llvm
                else:
                    boxed_arg = self._box(arg_val)
            else:
                # It's already an LLVM IR value
                if hasattr(arg_val, 'type') and arg_val.type != BOXED_PTR:
                    boxed_arg = self._box(Value(arg_val, PyType.OBJECT))
                else:
                    boxed_arg = arg_val

            field_ptr = self.builder.gep(
                args_ptr, [z, ir.Constant(I32, i)], inbounds=True
            )
            self.builder.store(boxed_arg, field_ptr)

        # Get the coroutine entry function pointer
        coro_entry_fn = self.functions.get(entry_name)
        if coro_entry_fn is None:
            raise CompileError(f"Brak funkcji coroutine entry: {entry_name}")

        # Cast function pointer to void(*)(void*) for __async_spawn
        fn_ptr_ty = ir.PointerType(ir.FunctionType(VOID, [I8P]))
        fn_ptr = self.builder.bitcast(coro_entry_fn, fn_ptr_ty, "coro_fn_ptr")

        # Cast args_ptr to i8* for __async_spawn
        args_i8p = self.builder.bitcast(args_ptr, I8P, "args_i8p")

        # Spawn the task: Task* __async_spawn(fn, arg)
        task_ptr = self.builder.call(
            self.functions["__async_spawn"],
            [fn_ptr, args_i8p],
            name="task"
        )

        # Box the Task* as Tag.TASK
        task_boxed = self._box_task(task_ptr)

        # Return as PyType.OBJECT (not PyType.TASK) so that variable
        # assignment and loading work correctly with the existing boxing
        # infrastructure.  The Tag.TASK in the boxed value is used for
        # runtime type dispatch in visit_Await and gather.
        return Value(task_boxed, PyType.OBJECT)

    def _box_task(self, task_ptr: ir.Value) -> ir.Value:
        """Box a Task* pointer as a BoxedValue with Tag.TASK.

        Creates: {GC_HEADER, tag=Tag.TASK, payload=ptrtoint(task_ptr, i64)}
        """
        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)], "task_box_raw")
        bv = self.builder.bitcast(raw, BOXED_PTR, "task_bv")
        z = ir.Constant(I32, 0)
        null_i8p = ir.Constant(I8P, None)

        # GC header
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
        # tag = Tag.TASK
        self.builder.store(
            ir.Constant(I64, Tag.TASK),
            self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True),
        )
        # payload = ptrtoint(task_ptr, i64)
        self.builder.store(
            self.builder.ptrtoint(task_ptr, I64),
            self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True),
        )

        return bv

    def _unbox_task(self, val: Value) -> ir.Value:
        """Unbox a Task* from a boxed value with Tag.TASK.

        Extracts the payload and converts back to i8* (Task*).
        """
        tag, payload = self._read_slot(val.llvm)
        return self.builder.inttoptr(payload, I8P, "task_ptr")

    # ── Static helper ──────────────────────────────────────────

    @staticmethod
    def get_async_runtime_c() -> str:
        """Return the C source code for the async runtime scheduler."""
        return ASYNC_RUNTIME_C

    def _has_async_functions(self) -> bool:
        """Check if any async functions were defined."""
        return bool(getattr(self, '_async_functions', set()))
