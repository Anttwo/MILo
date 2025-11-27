// force_abi.h
// -----------------------------------------------------------------------------
// Force GNU libstdc++ to use the *new* C++11 ABI (_GLIBCXX_USE_CXX11_ABI = 1)
// for this translation unit. **Must be included before ANY standard headers.**
//
// Why?
// - PyTorch ≥ 2.6 official binaries are built with the new C++11 ABI (=1).
// - If your extension is compiled with the old ABI (=0), you’ll hit runtime
//   linker errors such as undefined symbol: c10::detail::torchCheckFail(...RKSs).
//
// Usage:
//   Place this as the VERY FIRST include in each .cpp that builds your extension:
//     #include "force_abi.h"
//     #include <pybind11/pybind11.h>
//     ...
// -----------------------------------------------------------------------------

#pragma once

// Only meaningful for GCC's libstdc++; harmless elsewhere.
#if defined(__GNUC__) && !defined(_LIBCPP_VERSION)

// If the macro was already defined (e.g., by compiler flags), reset it first.
#  ifdef _GLIBCXX_USE_CXX11_ABI
#    undef _GLIBCXX_USE_CXX11_ABI
#  endif
// Enforce the new (C++11) ABI.
#  define _GLIBCXX_USE_CXX11_ABI 1

// Optional sanity check: if some libstdc++ internals are already visible,
// it likely means a standard header slipped in before this file. In that case
// overriding the ABI here won't affect those already-included headers.
#  if defined(_GLIBCXX_RELEASE) || defined(__GLIBCXX__) || defined(_GLIBCXX_BEGIN_NAMESPACE_VERSION)
#    warning "force_abi.h should be included BEFORE any standard library headers."
#  endif

#endif  // defined(__GNUC__) && !defined(_LIBCPP_VERSION)

