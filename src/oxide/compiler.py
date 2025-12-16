"""Compiler pipeline for the Vibec language."""

import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass

from .lexer import LexerError, tokenize
from .parser import ParseError, parse
from .checker import TypeError, check
from .codegen import generate


@dataclass
class CompileResult:
  """Result of a compilation."""

  success: bool
  error: str | None = None
  assembly: str | None = None


class Compiler:
  """Orchestrates the compilation pipeline."""

  def compile_to_asm(self, source: str) -> CompileResult:
    """Compile source code to ARM64 assembly."""
    try:
      # Lexing
      tokens = tokenize(source)

      # Parsing
      ast = parse(tokens)

      # Type checking (also transforms AST for type inference)
      transformed_ast, type_check_result = check(ast)

      # Code generation (uses transformed AST with resolved generic calls)
      assembly = generate(transformed_ast, type_check_result)

      return CompileResult(success=True, assembly=assembly)

    except LexerError as e:
      return CompileResult(success=False, error=f"Lexer error: {e}")
    except ParseError as e:
      return CompileResult(success=False, error=f"Parse error: {e}")
    except TypeError as e:
      return CompileResult(success=False, error=f"Type error: {e}")
    except Exception as e:
      return CompileResult(success=False, error=f"Internal error: {e}")

  def compile_to_binary(self, source: str, output_path: Path, keep_asm: bool = False) -> CompileResult:
    """Compile source code to an executable binary."""
    result = self.compile_to_asm(source)
    if not result.success or result.assembly is None:
      return result

    try:
      with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        asm_path = tmpdir_path / "output.s"
        obj_path = tmpdir_path / "output.o"

        # Write assembly
        asm_path.write_text(result.assembly)

        # Optionally save assembly file alongside output
        if keep_asm:
          asm_output = output_path.with_suffix(".s")
          asm_output.write_text(result.assembly)

        # Assemble
        assemble_result = subprocess.run(
          ["as", "-o", str(obj_path), str(asm_path)],
          capture_output=True,
          text=True,
        )
        if assemble_result.returncode != 0:
          return CompileResult(
            success=False,
            error=f"Assembler error: {assemble_result.stderr}",
          )

        # Link
        link_result = subprocess.run(
          [
            "ld",
            "-o",
            str(output_path),
            str(obj_path),
            "-lSystem",
            "-syslibroot",
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
            "-e",
            "_main",
            "-arch",
            "arm64",
          ],
          capture_output=True,
          text=True,
        )
        if link_result.returncode != 0:
          return CompileResult(success=False, error=f"Linker error: {link_result.stderr}")

      return CompileResult(success=True, assembly=result.assembly)

    except FileNotFoundError as e:
      return CompileResult(
        success=False,
        error=f"Tool not found: {e}. Make sure Xcode Command Line Tools are installed.",
      )
    except Exception as e:
      return CompileResult(success=False, error=f"Build error: {e}")


def compile_source(source: str) -> CompileResult:
  """Convenience function to compile source to assembly."""
  return Compiler().compile_to_asm(source)


def compile_file(source_path: Path, output_path: Path, keep_asm: bool = False) -> CompileResult:
  """Compile a source file to an executable."""
  source = source_path.read_text()
  return Compiler().compile_to_binary(source, output_path, keep_asm)
