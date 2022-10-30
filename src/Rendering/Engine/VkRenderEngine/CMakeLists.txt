set(LIB_NAME VkRenderEngine)

file(
    GLOB_RECURSE LIB_SRC 
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c*"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
)

add_library(${LIB_NAME} STATIC ${LIB_SRC})

add_compile_definitions(_ENABLE_EXTENDED_ALIGNED_STORAGE)

if(WIN32)
    foreach(SRC IN ITEMS ${LIB_SRC})
        get_filename_component(SRC_PATH "${SRC}" PATH)
        file(RELATIVE_PATH SRC_PATH_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${SRC_PATH}")
        string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
        source_group("${GROUP_PATH}" FILES "${SRC}")
    endforeach()
endif()

file(RELATIVE_PATH PROJECT_PATH_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Rendering/Engine")

set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "px${LIB_NAME}-${PHYSIKA_LIBRARY_VERSION}")

target_link_libraries(${LIB_NAME} PUBLIC Core IO Framework Topology)

target_include_directories(${LIB_NAME} PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/Rendering/Engine/VkRenderEngine>)
