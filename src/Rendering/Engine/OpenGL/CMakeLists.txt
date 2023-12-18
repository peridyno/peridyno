set(LIB_NAME GLRenderEngine)

# shaders
set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shader)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shader)

file(GLOB_RECURSE SHADER_FILES 
	${SHADER_SOURCE_DIR}/*.glsl 
	${SHADER_SOURCE_DIR}/*.vert 
	${SHADER_SOURCE_DIR}/*.frag 
	${SHADER_SOURCE_DIR}/*.geom)
source_group("Shader Sources" FILES ${SHADER_FILES})

# use glslang for compling shaders to binary
# vulkan backend already include glslang
if(NOT ${PERIDYNO_GPU_BACKEND} STREQUAL "Vulkan")
    add_subdirectory(${PERIDYNO_ROOT}/external/glslang "${CMAKE_BINARY_DIR}/external")
endif()

set(SHADER_BINARY_FILES "")
foreach(_shader_src ${SHADER_FILES})
    get_filename_component(_file_ext ${_shader_src} LAST_EXT)

    # skip .glsl files...
    if(NOT ${_file_ext} STREQUAL ".glsl")
        get_filename_component(_shader_name ${_shader_src} NAME)
        get_filename_component(_shader_root ${_shader_src} DIRECTORY)
        # variable name
        string(REPLACE "." "_" VAR_NAME ${_shader_name})
        string(TOUPPER ${VAR_NAME} VAR_NAME)

        set(_shader_bin "${SHADER_BINARY_DIR}/${_shader_name}.h")
        set(_cmd "$<TARGET_FILE:glslangValidator>" --target-env opengl --vn ${VAR_NAME} ${_shader_src} -o ${_shader_bin})
        
        add_custom_command(
            OUTPUT ${_shader_bin}
            #COMMAND echo ${_cmd}
            COMMAND ${_cmd}
            WORKING_DIRECTORY ${_shader_root}
            MAIN_DEPENDENCY ${_shader_src}
            DEPENDS glslangValidator 
            VERBATIM
        )
        list(APPEND SHADER_BINARY_FILES ${_shader_bin})        
    endif()
endforeach()
source_group("Shader Binaries" FILES ${SHADER_BINARY_FILES})


file(GLOB_RECURSE SOURCES *.cpp *.h *.c *.cu)

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "NoGPU")
    list(FILTER SOURCES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/Backend/*.*")
elseif("${PERIDYNO_GPU_BACKEND}" STREQUAL "Vulkan")
    list(FILTER SOURCES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/Backend/Cuda/*.*")
elseif("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
    list(FILTER SOURCES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/Backend/Vulkan/*.*")
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
add_library(${LIB_NAME} SHARED ${SOURCES} ${SHADER_FILES} ${SHADER_BINARY_FILES})

# for binary shader data
target_include_directories(${LIB_NAME} PRIVATE ${SHADER_BINARY_DIR})
#add_dependencies(${LIB_NAME} CompileShaders)

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "NoGPU")
    target_link_libraries(${LIB_NAME} PUBLIC
	    Core 
	    Framework 
	    RenderCore 
	    glad 
	    imgui)
else()
    target_link_libraries(${LIB_NAME} PUBLIC
	    Core 
	    Framework 
	    Topology
	    RenderCore 
	    glad 
	    imgui)
endif()

target_include_directories(${LIB_NAME} 
	PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
)

#To disable the warning "calling a constexpr __host__ function("***") from a __host__ __device__ function("***") is not allowed."
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

#Tell the required compilers to enable device linking on the library target.
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Rendering/Engine")
set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "dyno${LIB_NAME}-${PERIDYNO_LIBRARY_VERSION}")

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_FLAGS}")
endif()

add_compile_definitions(PERIDYNO_API_EXPORTS)

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

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "NoGPU")
    target_include_directories(${LIB_NAME} PUBLIC
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Module>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Node>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL>)
elseif("${PERIDYNO_GPU_BACKEND}" STREQUAL "Vulkan")
    target_include_directories(${LIB_NAME} PUBLIC
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Module>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Node>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Backend>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Backend/Vulkan>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Backend/Vulkan/Module>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL>)
elseif("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
    #To resolve the error: Target "..." INTERFACE_INCLUDE_DIRECTORIES property contains path: "..." which is prefixed in the build directory.
    target_include_directories(${LIB_NAME} PUBLIC
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Module>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Node>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Backend>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Backend/Cuda>
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/Engine/OpenGL/Backend/Cuda/Module>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/Module>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/Backend/Cuda>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/Backend/Cuda/Module>)
endif()


#add_dependencies(${LIB_NAME} BuildShaderHeader)
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

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL)

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Module/*.h")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/Module)

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/gl/*.h")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/gl)

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/shader/*.*")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/shader)

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/shader/helper/*.*")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/shader/helpler)

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
    file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Backend/Cuda/Module/*.h")
    install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/Backend/Cuda/Module)

    file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Backend/Cuda/Node/*.h")
    install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/Backend/Cuda/Node)
endif()

