
# 🚀 PyLow Compiler

**PyLow** is an ambitious compiler project aimed at completely eliminating the trade-off between the ease of writing Python and the performance of low-level languages.

PyLow is not just another interpreter or virtual layer. It is a tool that transforms Python code directly into high-performance machine code using the **LLVM** infrastructure.

## 🌟 Project Vision

PyLow is built on three pillars:

1. **100% Compatibility:** Our ultimate goal is full compatibility with the Python language standard. Your code should work exactly the same way, whether you run it in the standard interpreter or compile it with PyLow.
2. **Extreme Speed:** By leveraging LLVM, PyLow optimizes code at the CPU instruction level, eliminating overhead typical of virtual machines.
3. **Lightweight:** Designed with minimal resource consumption in mind. PyLow aims for the final binary files to be small, fast to launch, and independent of heavy runtimes.

## ⚙️ Compilation Architecture

The code transformation process in PyLow runs in a linear and optimized manner:

`Source Code (.py)` > `Analysis and Optimization` > `LLVM IR (.ll)` > `Native Code (Binary)`

By generating `.ll` (LLVM Intermediate Representation) intermediary files, PyLow can utilize the powerful LLVM optimizers used by the fastest compilers in the world (such as Clang or Rustc).

## 🛠️ Installation

Installation has been streamlined into a single, smart command that configures the entire environment for you.

### 🐧 Linux / 🍎 macOS

```bash
bash -c "$(curl -sSL https://raw.githubusercontent.com/pp1sp1/pylow/main/install.sh)"

```

### 🪟 Windows (PowerShell)

```powershell
& (iwr -useb https://raw.githubusercontent.com/pp1sp1/pylow/main/install.ps1)

```

*The installer will automatically download the latest version from **GitHub Releases**, install the necessary dependencies (LLVM, CMake, Git), and add the `pylow` command to your system path.*

## 🚀 Usage

After installation, the compiler is available globally in your terminal:

```bash
pylow your_file.py

```

## ⚠️ Project Status: Alpha Phase

**NOTE:** PyLow is currently in an **intensive development phase (Alpha)**. This is a bold, new project, which means that:

* 🛠️ **Under construction:** Not all Python language features are fully implemented yet.
* 🧪 **Testing:** You may encounter bugs during the compilation process or code generation.
* 📈 **Evolution:** The architecture may undergo changes to improve performance.

If you want to help develop PyLow, please report bugs in the *Issues* section or submit your proposed changes via *Pull Requests*.
