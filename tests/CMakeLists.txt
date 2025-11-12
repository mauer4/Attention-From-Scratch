add_executable(test_placeholder test_placeholder.cpp)
target_link_libraries(test_placeholder attention_core)

# Validates safetensors + config plumbing for the basic custom engine.
add_executable(test_loader test_loader.cpp)
target_link_libraries(test_loader attention_core)
target_compile_definitions(test_loader PRIVATE PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME test_loader COMMAND test_loader)

add_executable(test_prefill_decode test_prefill_decode.cpp)
target_include_directories(test_prefill_decode PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_prefill_decode attention_core CUDA::cudart)
add_test(NAME test_prefill_decode COMMAND test_prefill_decode)

add_executable(test_layer0_smoke test_layer0_smoke.cpp)
target_link_libraries(test_layer0_smoke attention_core CUDA::cudart)
add_test(NAME test_layer0_smoke COMMAND test_layer0_smoke --verbose)
