set(LIB_NAME Core)

set(LIB_SRC_DIR "${PROJECT_SOURCE_DIR}/src/${LIB_NAME}")
file(
    GLOB_RECURSE LIB_SRC 
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${LIB_SRC_DIR}/*.c*"
    "${LIB_SRC_DIR}/*.h*"
	"${LIB_SRC_DIR}/*.inl"
)

if(${LIB_NAME} STREQUAL "Core")
    list(APPEND LIB_SRC "${PROJECT_SOURCE_DIR}/external/tinyxml/tinyxml2.cpp")
    list(FILTER LIB_SRC EXCLUDE REGEX Platform.h.in)
endif()

if(WIN32)
    add_library(${LIB_NAME} STATIC ${LIB_SRC})
elseif(UNIX)
    add_library(${LIB_NAME} SHARED ${LIB_SRC}) 
endif()

if(WIN32)
    foreach(SRC IN ITEMS ${LIB_SRC})
        get_filename_component(SRC_PATH "${SRC}" PATH)
        file(RELATIVE_PATH SRC_PATH_REL "${LIB_SRC_DIR}" "${SRC_PATH}")
        string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
        source_group("${GROUP_PATH}" FILES "${SRC}")
    endforeach()
endif()

if(WIN32)
    target_compile_options(${LIB_NAME} PRIVATE -Xcompiler "/wd 4819") 
endif()

file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine")
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${LIB_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_FLAGS}")
set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "dyno${LIB_NAME}-${PERIDYNO_LIBRARY_VERSION}")

#To resolve the error: Target "..." INTERFACE_INCLUDE_DIRECTORIES property contains path: "..." which is prefixed in the build directory.
#    target_include_directories(${LIB_NAME} PUBLIC ${PERIDYNO_INCLUDE_PATH}) # for generated files
target_include_directories(${LIB_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/Core>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/eigen>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/glm-0.9.9.7>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/external/glm-0.9.9.7>)

#To disable the warning "calling a constexpr __host__ function("***") from a __host__ __device__ function("***") is not allowed."
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

install(TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}Targets
    RUNTIME  DESTINATION  ${PERIDYNO_RUNTIME_INSTALL_DIR}
    LIBRARY  DESTINATION  ${PERIDYNO_LIBRARY_INSTALL_DIR}
    ARCHIVE  DESTINATION  ${PERIDYNO_ARCHIVE_INSTALL_DIR}
    )

install(EXPORT ${LIB_NAME}Targets DESTINATION ${PERIDYNO_CMAKE_CONFIG_INSTALL_DIR}
    FILE ${LIB_NAME}Targets.cmake)

get_property(LOCAL_CMAKES_NAMES GLOBAL PROPERTY "GLOBAL_CMAKES_NAMES")
list(APPEND LOCAL_CMAKES_NAMES "${LIB_NAME}Targets.cmake")    
set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES ${LOCAL_CMAKES_NAMES})

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.inl")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core)

file(GLOB CORE_ALGORITHM_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Algorithm/*.h")
install(FILES ${CORE_ALGORITHM_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/Algorithm)

file(GLOB CORE_VECTOR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Vector/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/Vector/*.inl")
install(FILES ${CORE_VECTOR_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/Vector)

file(GLOB CORE_MATRIX_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Matrix/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/Matrix/*.inl")
install(FILES ${CORE_MATRIX_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/Matrix)

file(GLOB CORE_RIGID_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Rigid/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/Rigid/*.inl")
install(FILES ${CORE_RIGID_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/Rigid)

file(GLOB CORE_ARRAY_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Array/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/Array/*.inl")
install(FILES ${CORE_ARRAY_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/Array)

file(GLOB CORE_ARRAY_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/STL/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/STL/*.inl")
install(FILES ${CORE_ARRAY_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/STL)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Platform.h.in" "${CMAKE_CURRENT_SOURCE_DIR}/Platform.h")

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/Platform.h"  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/external/glm-0.9.9.7/" DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/external/glm-0.9.9.7/)