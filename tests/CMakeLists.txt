# Gizmo automated tests
#
# Build a small in-process test runner. Pure unit tests (CLI parser,
# context-size math) run without a model. Engine integration tests are
# skipped at runtime unless GIZMO_TEST_MODEL points at an existing GGUF.

set(GIZMO_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_context_math.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cli_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_engine_context.cpp
)

add_executable(gizmo_tests ${GIZMO_TEST_SOURCES})

target_include_directories(gizmo_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_link_libraries(gizmo_tests PRIVATE gizmo_core)

# Optional model fixture for engine integration tests.
if(DEFINED ENV{GIZMO_TEST_MODEL})
    set(GIZMO_TEST_MODEL "$ENV{GIZMO_TEST_MODEL}")
endif()

add_test(NAME gizmo_unit_and_integration_tests
    COMMAND gizmo_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

if(GIZMO_TEST_MODEL)
    set_property(TEST gizmo_unit_and_integration_tests PROPERTY
        ENVIRONMENT "GIZMO_TEST_MODEL=${GIZMO_TEST_MODEL}")
endif()

# Smoke-test the built gizmo binary. Runs in the build tree; the script skips
# gracefully when the binary is not present.
add_test(NAME gizmo_cli_smoke_tests
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/integration.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Point the smoke test at the gizmo binary produced in the parent build dir.
set_property(TEST gizmo_cli_smoke_tests PROPERTY
    ENVIRONMENT "GIZMO_BIN=$<TARGET_FILE:gizmo>")
