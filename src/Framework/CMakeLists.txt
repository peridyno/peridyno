set(LIB_NAME Framework)

set(LIB_SRC_DIR "${PROJECT_SOURCE_DIR}/src/${LIB_NAME}")
file(                                                                           #利用glob命令读取所有源文件list
    GLOB_RECURSE LIB_SRC 
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${LIB_SRC_DIR}/*.c*"
    "${LIB_SRC_DIR}/*.h*"
	"${LIB_SRC_DIR}/*.inl"
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

if(WIN32)
    target_compile_options(${LIB_NAME} PRIVATE -Xcompiler "/wd 4819") 
endif()

file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine")
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${LIB_NAME} PROPERTIES CUDA_ARCHITECTURES "60")
 
set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "dyno${LIB_NAME}-${PERIDYNO_LIBRARY_VERSION}")
set_target_properties(${LIB_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

#To resolve the error: Target "..." INTERFACE_INCLUDE_DIRECTORIES property contains path: "..." which is prefixed in the build directory.
#    target_include_directories(${LIB_NAME} PUBLIC ${PERIDYNO_INCLUDE_PATH}) # for generated files
target_include_directories(${LIB_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/Framework>
    $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/filesystem>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME}>
    #$<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/external/filesystem>
	)

#To disable the warning "calling a constexpr __host__ function("***") from a __host__ __device__ function("***") is not allowed."
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_link_libraries(${LIB_NAME} Core)

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


file(GLOB FRAMEWORK_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.inl")
install(FILES ${FRAMEWORK_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Framework)

file(GLOB ACTION_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Action/*.h")
install(FILES ${ACTION_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Framework/Action)

file(GLOB COLLISION_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Collision/*.h")
install(FILES ${COLLISION_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Framework/Collision)

file(GLOB FRAMEWORK_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Framework/*.h")
install(FILES ${FRAMEWORK_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Framework)

file(GLOB FRAMEWORK_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Module/*.h")
install(FILES ${FRAMEWORK_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Module)

file(GLOB MAPPING_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Mapping/*.h")
install(FILES ${MAPPING_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Framework/Mapping)

file(GLOB TOPOLOGY_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Topology/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/Topology/*.inl")
install(FILES ${TOPOLOGY_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Framework/Topology)