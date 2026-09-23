#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$base/build"
compiler=${CXX:-clang++-18}
llvm_config=${LLVM_CONFIG:-llvm-config-18}
objects=()
for source in "$base"/src/*.cpp "$base"/cupbop/src/*.cpp; do
  object="$base/build/$(basename "$source" .cpp).o"
  rebuild=0
  test -f "$object" && test "$object" -nt "$source" || rebuild=1
  test "$object" -nt "$base/build.sh" || rebuild=1
  for header in "$base"/cupbop/include/*.h "$base"/src/*.h; do
    test "$object" -nt "$header" || rebuild=1
  done
  if [ "$rebuild" = 1 ]; then
    "$compiler" $("$llvm_config" --cxxflags) -std=c++17 -O1 -g -fexceptions \
      -I"$base/cupbop/include" -c "$source" -o "$object"
  fi
  objects+=("$object")
done
"$compiler" "${objects[@]}" $("$llvm_config" --ldflags --system-libs --libs all) -o "$base/build/cpu-coarsen"
