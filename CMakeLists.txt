cmake_minimum_required(VERSION 3.18) # Wymagane dla obsługi CUDA w CMake
project(MyCudaProject LANGUAGES CXX CUDA)

cmake_minimum_required(VERSION 3.18) # Wymagane dla CUDA i nowoczesnego CMake
project(KMeans LANGUAGES CXX CUDA)

# Ustaw wersję standardu C++ i CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Źródła projektu
set(CPP_SOURCES
    main.cpp
    display.cpp
)

set(CUDA_SOURCES
    data_generator.cu
)

# Dodaj katalogi z nagłówkami
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Znajdź SDL2
find_package(SDL2 REQUIRED)

# Dodaj katalogi SDL2
include_directories(${SDL2_INCLUDE_DIRS})

# Utwórz plik wykonywalny z plików C++ i CUDA
add_executable(KMeans ${CPP_SOURCES} ${CUDA_SOURCES})

# Linkowanie SDL2
target_link_libraries(KMeans PRIVATE ${SDL2_LIBRARIES})

# Ustawienie flag kompilatora dla CUDA (opcjonalne)
set_target_properties(KMeans PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON # Umożliwia oddzielną kompilację plików CUDA
)

# Opcjonalne flagi do debugowania i optymalizacji
target_compile_options(KMeans PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
