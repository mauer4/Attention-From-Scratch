set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_placeholder.cpp
)

add_executable(attention_tests ${TEST_SOURCES})
target_link_libraries(attention_tests PRIVATE attention_core)

add_test(NAME attention_placeholder COMMAND attention_tests)
