#!/usr/bin/env bash
set -euo pipefail

LAMMPS_ROOT="${LAMMPS_ROOT:-/home/star/Research/software/lammps-22Jul2025}"
BUILD_DIR="${BUILD_DIR:-$LAMMPS_ROOT/build_openmp}"
JOBS="${JOBS:-$(nproc)}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [[ ! -d "$LAMMPS_ROOT/cmake" ]]; then
  echo "Cannot find LAMMPS CMake source directory: $LAMMPS_ROOT/cmake" >&2
  echo "Set LAMMPS_ROOT to the LAMMPS source root." >&2
  exit 2
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "Cannot find cmake in PATH." >&2
  exit 2
fi

echo "LAMMPS root : $LAMMPS_ROOT"
echo "Build dir   : $BUILD_DIR"
echo "Jobs        : $JOBS"
echo "Build type  : $CMAKE_BUILD_TYPE"
echo
echo "This creates a separate build and does not overwrite $LAMMPS_ROOT/build/lmp."

cmake -S "$LAMMPS_ROOT/cmake" -B "$BUILD_DIR" \
  -D CMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -D BUILD_MPI=on \
  -D BUILD_OMP=on \
  -D PKG_MOLECULE=on \
  -D PKG_ASPHERE=on \
  -D PKG_RIGID=on \
  -D PKG_OPENMP=on

cmake --build "$BUILD_DIR" -j "$JOBS"

LMP="$BUILD_DIR/lmp"
if [[ ! -x "$LMP" ]]; then
  echo "Build finished but executable is missing: $LMP" >&2
  exit 1
fi

echo
echo "Built LAMMPS executable:"
echo "$LMP"
echo
echo "Required package check:"
missing=0
for package in MOLECULE ASPHERE RIGID OPENMP; do
  if "$LMP" -h | tr -cs '[:alnum:]_' '\n' | grep -Fxq "$package"; then
    echo "  $package: OK"
  else
    echo "  $package: MISSING"
    missing=1
  fi
done

if (( missing != 0 )); then
  echo "One or more required packages are missing from $LMP." >&2
  exit 1
fi

echo
echo "Use it with:"
echo "  LAMMPS_BIN=$LMP"
