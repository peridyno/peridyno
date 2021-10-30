set(LIB_NAME ABCExporter)

set(LIB_SRC_DIR "${PROJECT_SOURCE_DIR}/src/${LIB_NAME}")
file(
    GLOB_RECURSE LIB_SRC 
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${LIB_SRC_DIR}/*.c*"
    "${LIB_SRC_DIR}/*.h*"
)

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

file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine")

set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "dyno${LIB_NAME}-${PERIDYNO_LIBRARY_VERSION}")
set_target_properties(${LIB_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

#To resolve the error: Target "..." INTERFACE_INCLUDE_DIRECTORIES property contains path: "..." which is prefixed in the build directory.
#    target_include_directories(${LIB_NAME} PUBLIC ${PERIDYNO_INCLUDE_PATH}) # for generated files
target_include_directories(${LIB_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/ABCExporter>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME}>)

target_link_libraries(${LIB_NAME} Core Framework Alembic)