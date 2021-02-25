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

if(${LIB_NAME} STREQUAL "IO")
    list(APPEND LIB_SRC "${PROJECT_SOURCE_DIR}/external/LodePNG/lodepng.cpp")
endif()

if(WIN32)
    add_library(${LIB_NAME} STATIC ${LIB_SRC})
elseif(UNIX)
    add_library(${LIB_NAME} SHARED ${LIB_SRC}) 
endif()

if(WIN32)
    foreach(SRC IN ITEMS ${LIB_SRC})                                                #为VS工程添加filter 方便查看文件结构目录
        get_filename_component(SRC_PATH "${SRC}" PATH)
        file(RELATIVE_PATH SRC_PATH_REL "${LIB_SRC_DIR}" "${SRC_PATH}")
        string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
        source_group("${GROUP_PATH}" FILES "${SRC}")
    endforeach()
endif()

if(WIN32)                                                                       #only for windows compile option
    target_compile_options(${LIB_NAME} PRIVATE -Xcompiler "/wd 4819")               #禁止编译时报告文件编码不是unicode的warning，由于cuda头文件都不是unicode。使编译报错更清晰
endif()

file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                 #判断当前project在根目录下的相对路径
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine")      #为project设定folder目录
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)    #cmake 会自动延迟cuda dlink操作直到静态库被编译成动态库或者exe，
                                                                                #但这样影响了后面其他工程对PhysLeo.lib的引用，现在还有些原因未明，暂时在编译成库之前开启dlink
set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "dyno${LIB_NAME}-${PERIDYNO_LIBRARY_VERSION}")

#To resolve the error: Target "..." INTERFACE_INCLUDE_DIRECTORIES property contains path: "..." which is prefixed in the build directory.
#    target_include_directories(${LIB_NAME} PUBLIC ${PERIDYNO_INCLUDE_PATH}) # for generated files
target_include_directories(${LIB_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/Core>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/glm-0.9.9.7>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/external/glm-0.9.9.7>)

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
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core)

file(GLOB CORE_ALGORITHM_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Algorithm/*.h")
install(FILES ${CORE_ALGORITHM_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Algorithm)

file(GLOB CORE_VECTOR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Vector/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/Vector/*.inl")
install(FILES ${CORE_VECTOR_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/Vector)

file(GLOB CORE_MATRIX_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Matrix/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/Matrix/*.inl")
install(FILES ${CORE_MATRIX_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/Matrix)

file(GLOB CORE_RIGID_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Rigid/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/Rigid/*.inl")
install(FILES ${CORE_RIGID_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/Rigid)

file(GLOB CORE_ARRAY_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Array/*.h")
install(FILES ${CORE_ARRAY_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/Array)

file(GLOB CORE_UTILITY_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Utility/*.h")
install(FILES ${CORE_UTILITY_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/Utility)

file(GLOB CORE_QUAT_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Quaternion/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/Quaternion/*.inl")
install(FILES ${CORE_QUAT_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Quaternion)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Platform.h.in" "${CMAKE_CURRENT_SOURCE_DIR}/Platform.h")

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/Platform.h"  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Core/)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/external/glm-0.9.9.7/" DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/external/glm-0.9.9.7/)