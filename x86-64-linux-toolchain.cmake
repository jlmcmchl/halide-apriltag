# CMake toolchain file for x86-64-linux (host architecture)
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=x86-64-linux-toolchain.cmake ..

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Use clang-20 compilers
set(CMAKE_CXX_COMPILER "clang++-20")
set(CMAKE_C_COMPILER "clang-20")

# Set libc++ compile and link flags (for clang compatibility)
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -stdlib=libc++ -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-stdlib=libc++ -lc++abi")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-stdlib=libc++ -lc++abi")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-stdlib=libc++ -lc++abi")

