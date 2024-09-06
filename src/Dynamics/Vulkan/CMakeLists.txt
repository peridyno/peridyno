cmake_minimum_required(VERSION 3.10)

cmake_minimum_required(VERSION 3.10)

#foreach(LIB_NAME IN ITEMS ${LIB_NAMES})
macro(build_library LIB_NAME LIB_DEPENDENCY)
    set(LIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}")

    file(                                                                           
        GLOB_RECURSE LIB_SRC
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${LIB_SRC_DIR}/*.c*"
        "${LIB_SRC_DIR}/*.h*"
    )

    add_library(${LIB_NAME} SHARED ${LIB_SRC}) 

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
    set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine/Dynamics")

    set_target_properties(${LIB_NAME} PROPERTIES
        OUTPUT_NAME "px${LIB_NAME}-${PHOENIX_PHYSICS_VERSION}")

    add_compile_definitions(PERIDYNO_API_EXPORTS)
    add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)
	add_compile_definitions(_ENABLE_EXTENDED_ALIGNED_STORAGE)

    if(WIN32)
        set_target_properties(${LIB_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    elseif(UNIX)
        if (CMAKE_BUILD_TYPE MATCHES Debug)
            set_target_properties(${LIB_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
        else()
            set_target_properties(${LIB_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
        endif()
    endif()

    target_include_directories(${LIB_NAME} PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/Dynamics/Vulkan>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/src/Dynamics/Vulkan>)

    target_link_libraries(${LIB_NAME} PUBLIC ${${LIB_DEPENDENCY}})
endmacro()
       
option(PERIDYNO_LIBRARY_PARTICLESYSTEM "Enable binding the particle system library" ON)

if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY Core Framework Topology IO GLRenderEngine SPIRV glslang)
    build_library(ParticleSystem LIB_DEPENDENCY)
endif()