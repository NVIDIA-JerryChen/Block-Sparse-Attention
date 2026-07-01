#pragma once

// Compile-time dispatch helper: at runtime picks a branch, inside each branch
// CONST_NAME is a compile-time `constexpr bool` usable in template arguments.
//
// Usage:
//   BOOL_SWITCH(cond, HAS_X, [&] { run<HAS_X>(...); });
//
// For multiple axes, nest the macros:
//   BOOL_SWITCH(c1, A, [&] {
//     BOOL_SWITCH(c2, B, [&] { run<A, B>(...); });
//   });
#define BOOL_SWITCH(COND, CONST_NAME, ...)                \
    [&] {                                                 \
        if (COND) {                                       \
            constexpr static bool CONST_NAME = true;      \
            return __VA_ARGS__();                         \
        } else {                                          \
            constexpr static bool CONST_NAME = false;     \
            return __VA_ARGS__();                         \
        }                                                 \
    }()
