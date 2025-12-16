"""Command-line interface for the Oxide compiler."""

import sys
import argparse
from pathlib import Path

from .compiler import Compiler


def main() -> int:
  """Main entry point for the oxide compiler."""
  parser = argparse.ArgumentParser(
    prog="oxide",
    description="Oxide compiler - A Python/Rust hybrid language for ARM64 macOS",
  )
  parser.add_argument("source", type=Path, help="Source file to compile (.ox)")
  parser.add_argument("-o", "--output", type=Path, help="Output file (default: source name without extension)")
  parser.add_argument(
    "--emit-asm",
    action="store_true",
    help="Output assembly instead of compiling to binary",
  )
  parser.add_argument(
    "--keep-asm",
    action="store_true",
    help="Keep assembly file alongside binary",
  )

  args = parser.parse_args()

  # Validate source file
  if not args.source.exists():
    print(f"Error: Source file '{args.source}' not found", file=sys.stderr)
    return 1

  # Determine output path
  if args.output:
    output_path = args.output
  else:
    output_path = args.source.with_suffix("")

  # Read source
  source = args.source.read_text()

  # Compile
  compiler = Compiler()

  if args.emit_asm:
    result = compiler.compile_to_asm(source)
    if result.success:
      if args.output:
        args.output.write_text(result.assembly)
      else:
        print(result.assembly)
      return 0
    else:
      print(f"Error: {result.error}", file=sys.stderr)
      return 1
  else:
    result = compiler.compile_to_binary(source, output_path, keep_asm=args.keep_asm)
    if result.success:
      print(f"Compiled to {output_path}")
      return 0
    else:
      print(f"Error: {result.error}", file=sys.stderr)
      return 1


if __name__ == "__main__":
  sys.exit(main())
