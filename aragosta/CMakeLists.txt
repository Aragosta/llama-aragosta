# aragosta: serving-oriented additions to llama.cpp. Hooked in from the top-level CMakeLists.txt.
if (LLAMA_BUILD_TOOLS OR LLAMA_BUILD_TESTS)
    add_executable(mmq-sweep tools/mmq_sweep.cpp)
    target_link_libraries(mmq-sweep PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(mmq-sweep PRIVATE cxx_std_17)
endif()
