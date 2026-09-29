#ifndef GIZMO_TEST_HARNESS_HPP
#define GIZMO_TEST_HARNESS_HPP

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace gizmo {
namespace test {

struct TestCase {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline int register_test(const char* name, std::function<void()> fn) {
    registry().push_back({name, std::move(fn)});
    return 0;
}

inline void check_true(bool value, const char* expr, const char* file, int line) {
    if (!value) {
        std::cerr << "ASSERT_TRUE failed: " << expr << " at " << file << ":" << line << "\n";
        std::abort();
    }
}

inline void check_false(bool value, const char* expr, const char* file, int line) {
    if (value) {
        std::cerr << "ASSERT_FALSE failed: " << expr << " at " << file << ":" << line << "\n";
        std::abort();
    }
}

// Convert a value to a string for diagnostic messages. Handles arithmetic
// types, enums (via their underlying value), and bools.
inline std::string value_to_string(const std::string& v) { return v; }
inline std::string value_to_string(const char* v) { return v ? std::string(v) : "(null)"; }

template <typename T>
inline std::string value_to_string(const T& v) {
    if constexpr (std::is_same_v<T, bool>) {
        return v ? "true" : "false";
    } else if constexpr (std::is_enum_v<T>) {
        return std::to_string(static_cast<std::underlying_type_t<T>>(v));
    } else if constexpr (std::is_arithmetic_v<T>) {
        return std::to_string(v);
    } else {
        std::ostringstream oss;
        oss << "(" << typeid(T).name() << " value)";
        return oss.str();
    }
}

template <typename T1, typename T2>
inline void check_eq(const T1& a, const T2& b, const char* aexpr, const char* bexpr, const char* file, int line) {
    if (!(a == b)) {
        std::cerr << "ASSERT_EQ failed: " << aexpr << " == " << bexpr
                  << " (" << value_to_string(a) << " != " << value_to_string(b)
                  << ") at " << file << ":" << line << "\n";
        std::abort();
    }
}

inline void check_float_eq(float a, float b, float tol, const char* aexpr, const char* bexpr, const char* file, int line) {
    if (std::fabs(a - b) > tol) {
        std::cerr << "ASSERT_FLOAT_EQ failed: " << aexpr << " == " << bexpr
                  << " (" << a << " != " << b << ") at " << file << ":" << line << "\n";
        std::abort();
    }
}

inline int run_all_tests() {
    const auto& tests = registry();
    std::size_t passed = 0;
    std::size_t skipped = 0;
    for (const auto& t : tests) {
        std::cout << "[RUN] " << t.name << "\n";
        try {
            t.fn();
            std::cout << "[PASS] " << t.name << "\n";
            ++passed;
        } catch (const std::exception& e) {
            std::cerr << "[FAIL] " << t.name << ": " << e.what() << "\n";
        }
    }
    std::cout << "\n" << passed << "/" << tests.size() << " tests passed"
              << (skipped > 0 ? ", " + std::to_string(skipped) + " skipped" : "")
              << "\n";
    return passed == tests.size() ? 0 : 1;
}

} // namespace test
} // namespace gizmo

#define TEST(name) \
    static void test_##name(); \
    static int gizmo_test_reg_##name = gizmo::test::register_test(#name, test_##name); \
    static void test_##name()

#define ASSERT_TRUE(expr) gizmo::test::check_true((expr), #expr, __FILE__, __LINE__)
#define ASSERT_FALSE(expr) gizmo::test::check_false((expr), #expr, __FILE__, __LINE__)
#define ASSERT_EQ(a, b) gizmo::test::check_eq((a), (b), #a, #b, __FILE__, __LINE__)
#define ASSERT_FLOAT_EQ_TOL(a, b, tol) gizmo::test::check_float_eq((a), (b), (tol), #a, #b, __FILE__, __LINE__)

#endif // GIZMO_TEST_HARNESS_HPP
