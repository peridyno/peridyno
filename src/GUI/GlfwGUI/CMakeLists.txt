cmake_minimum_required(VERSION 3.10)

set(LIB_NAME GlfwGUI)
set(LIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

file(                                                                           
    GLOB_RECURSE LIB_SRC
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
	"${LIB_SRC_DIR}/*.c*"
    "${LIB_SRC_DIR}/*.*"
)

if(WIN32)
    add_library(${LIB_NAME} STATIC ${LIB_SRC})
elseif(UNIX)
    add_library(${LIB_NAME} SHARED ${LIB_SRC}) 
endif()

foreach(SRC IN ITEMS ${LIB_SRC})
    get_filename_component(SRC_PATH "${SRC}" PATH)
    file(RELATIVE_PATH SRC_PATH_REL "${LIB_SRC_DIR}" "${SRC_PATH}")
    string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
    source_group("${GROUP_PATH}" FILES "${SRC}")
endforeach()

if(WIN32)
    target_compile_options(${LIB_NAME} PRIVATE -Xcompiler "/wd 4819")
endif()
file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine/GUI")
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "dyno${LIB_NAME}-${PERIDYNO_LIBRARY_VERSION}")

target_include_directories(${LIB_NAME} PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/GUI>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/GUI>)

target_link_libraries(${LIB_NAME} Core Framework IO RenderEngine glfw imgui)
