llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_ZSTD_SOURCES "")
if (LLAMA_ZSTD)
    set(LLAMA_ZSTD_SOURCES
        llama-weight-zstd.cpp
        ../vendor/zstd-seekable/zstdseek_compress.c
        ../vendor/zstd-seekable/zstdseek_decompress.c)
endif()

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            ${LLAMA_ZSTD_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (LLAMA_ZSTD)
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        pkg_check_modules(LIBZSTD libzstd)
    endif()
    if (LIBZSTD_FOUND)
        target_include_directories(llama PRIVATE ${LIBZSTD_INCLUDE_DIRS})
        target_link_libraries(llama PRIVATE ${LIBZSTD_LIBRARIES})
    else()
        find_library(ZSTD_LIB zstd REQUIRED)
        target_link_libraries(llama PRIVATE ${ZSTD_LIB})
    endif()
    # vendor/zstd-seekable provides its own mem.h stub; system headers supply xxhash.h
    target_include_directories(llama PRIVATE ../vendor/zstd-seekable /usr/include)
    target_compile_definitions(llama PUBLIC  GGML_USE_ZSTD)
    target_compile_definitions(ggml PUBLIC   GGML_USE_ZSTD)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
