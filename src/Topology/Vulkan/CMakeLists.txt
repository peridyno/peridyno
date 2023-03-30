set(LIB_NAME Topology)

file(                                                                      
    GLOB_RECURSE LIB_SRC 
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c*"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
)

add_library(${LIB_NAME} STATIC ${LIB_SRC})

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)
add_compile_definitions(_ENABLE_EXTENDED_ALIGNED_STORAGE)

if(WIN32)
    foreach(SRC IN ITEMS ${LIB_SRC})
        get_filename_component(SRC_PATH "${SRC}" PATH)
        file(RELATIVE_PATH SRC_PATH_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${SRC_PATH}")
        string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
        source_group("${GROUP_PATH}" FILES "${SRC}")
    endforeach()
endif()

file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}") 
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine")
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON) 

set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "px${LIB_NAME}-${PHOENIX_PHYSICS_VERSION}")

target_include_directories(${LIB_NAME} PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/Topology/Vulkan>)

target_link_libraries(${LIB_NAME} Core Framework)

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

file(GLOB COLLISION_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Collision/*.h")
install(FILES ${COLLISION_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Topology/Vulkan/Collision)

file(GLOB FRAMEWORK_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Topology/*.h")
install(FILES ${FRAMEWORK_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Topology/Vulkan/Topology)