set(LIB_NAME RenderCore)

file(GLOB_RECURSE SOURCES *.cpp *.h *.cu)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
add_library(${LIB_NAME} STATIC ${SOURCES})

target_link_libraries(${LIB_NAME} PRIVATE Core Framework)

target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/glm-0.9.9.7>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME}>)
	
#To disable the warning "calling a constexpr __host__ function("***") from a __host__ __device__ function("***") is not allowed."
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
set_target_properties(${LIB_NAME} PROPERTIES CUDA_ARCHITECTURES "60")

set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Rendering")