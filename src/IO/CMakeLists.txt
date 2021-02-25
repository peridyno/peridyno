set(LIB_NAME IO)

set(LIB_SRC_DIR "${PROJECT_SOURCE_DIR}/src/${LIB_NAME}")
file(                                                                           #利用glob命令读取所有源文件list
    GLOB_RECURSE LIB_SRC 
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${LIB_SRC_DIR}/*.c*"
    "${LIB_SRC_DIR}/*.h*"
)

list(FILTER LIB_SRC EXCLUDE REGEX .*deprecated/.*)                              #排除deprecated 目录下面的所有文件

if(${LIB_NAME} STREQUAL "Core")                                            #特殊处理 把stackWalker tinyxml2 lodepng3个源文件编入core模块
    if(WIN32)
        list(APPEND LIB_SRC "${PROJECT_SOURCE_DIR}/external/StackWalker/StackWalker.cpp")  #stackwalker is only for windows
    endif()
    list(APPEND LIB_SRC "${PROJECT_SOURCE_DIR}/external/tinyxml/tinyxml2.cpp")
    list(FILTER LIB_SRC EXCLUDE REGEX Platform.h.in)
endif()

if(${LIB_NAME} STREQUAL "IO")
    list(APPEND LIB_SRC "${PROJECT_SOURCE_DIR}/external/LodePNG/lodepng.cpp")        #为IO lib添加lodeong源文件
endif()

if(WIN32)
    add_library(${LIB_NAME} STATIC ${LIB_SRC})                                      #添加编译目标 静态链接库
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
set_target_properties(${LIB_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

#To resolve the error: Target "..." INTERFACE_INCLUDE_DIRECTORIES property contains path: "..." which is prefixed in the build directory.
#    target_include_directories(${LIB_NAME} PUBLIC ${PERIDYNO_INCLUDE_PATH}) # for generated files
target_include_directories(${LIB_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/IO>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME}>)

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

file(GLOB FILE_UTILITY_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/File_Utilities/*.h")
install(FILES ${FILE_UTILITY_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/IO/File_Utilities)

file(GLOB IMAGE_IO_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Image_IO/*.h")
install(FILES ${IMAGE_IO_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/IO/Image_IO)

file(GLOB SMESH_IO_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Smesh_IO/*.h")
install(FILES ${SMESH_IO_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/IO/Smesh_IO)

file(GLOB SURFACE_MESH_IO_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Surface_Mesh_IO/*.h")
install(FILES ${SURFACE_MESH_IO_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/IO/Surface_Mesh_IO)
