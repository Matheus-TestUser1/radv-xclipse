#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute the real ALU helper with SIMD primitives and check cache-key ABI.

This host adapter is not Mesa NIR, a backend compiler, or a GPU test.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
TESTS = Path(__file__).resolve().parent / "cooperative_matrix_alu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]

    header = (ROOT / "src/amd/vulkan/radv_shader.h").read_text()
    key = re.search(r"/\* Misc values included as part of the cache key \*/\s*struct \{(.*?)\}\s*key;",
                    header, re.S)
    if not key:
        raise RuntimeError("Compiler cache key declaration not found")
    cache_source = """
#include <stdint.h>
#include <string.h>
#include <assert.h>
struct key {""" + key[1] + """};
_Static_assert(sizeof(struct key) == 20, "Compiler cache key ABI changed");
int main(void)
{
    struct key keys[3];
    memset(keys, 0, sizeof(keys));
    keys[1].xclipse_coopmat_alu = 1;
    keys[2].xclipse_coopmat_alu = 2;
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = i + 1; j < 3; j++)
            assert(memcmp(&keys[i], &keys[j], sizeof(struct key)) != 0);
}
"""
    with tempfile.TemporaryDirectory(prefix="radv-coopmat-alu-") as directory:
        directory = Path(directory)
        cache = directory / "cache.c"
        cache.write_text(cache_source)
        cases = [(TESTS / "reference.c", "reference", ["-I" + str(TESTS / "adapter"), "-lm"]),
                 (cache, "cache", [])]
        for source, name, extra in cases:
            binary = directory / name
            subprocess.run(compiler + flags + [str(source)] + extra + ["-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
    print("PASS: 20-byte compiler cache key; disabled, legacy, and fixed-broadcast tags are distinct.")


if __name__ == "__main__":
    main()
