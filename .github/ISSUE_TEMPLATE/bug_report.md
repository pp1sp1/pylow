---
name: Bug report
about: Report a compiler bug (ICE, miscompilation, or incorrect error)
title: '[Component] Short description of the issue'
labels: 'bug'
assignees: ''

---

## 🐛 Bug Description
A clear and concise description of what the compiler bug is.
*(e.g., Compiler crashes/panics, generates incorrect target code, or rejects valid source code).*

---

## 💻 Environment Information
*Please provide details about the environment where the compiler is running.*

- **Compiler Version:** (Version or specific git commit hash)
- **Host OS & Architecture:** (The OS and architecture running the compiler)
- **Target OS & Architecture (if cross-compiling):** (The target platform you are generating code for)
- **Build System:** (The tool managing the project build process)

---

## 🛠️ Steps to Reproduce

### 1. Minimal Reproducible Example (MRE)
*Provide the smallest possible code snippet that triggers the bug. Use syntax highlighting.*

```python
# Paste the minimal source code that breaks the compiler here

```

### 2. Compilation Flags and Parameters

*Describe the options, optimization levels, or parameters passed to the compiler.*

---

## 📊 Actual Behavior vs Expected Behavior

### Actual Behavior

*What does the compiler currently do? Paste full error messages, stack traces, or incorrect intermediate/final output.*

```text
# Paste compiler error logs or incorrect output here

```

### Expected Behavior

*What should the compiler have done? (e.g., compiled successfully, generated a specific structure, raised a clean user-facing diagnostic error instead of crashing).*

---

## 🔍 Additional Context & Diagnostics

* **Is this a regression?** (Did this bug occur in previous versions of your code?)
* **Diagnostic Tool Outputs:** (Logs from internal compiler layers/passes, if available).
* Any other notes regarding the compiler phase (frontend, IR generation, optimizations) where you suspect the issue lies.
