cmake_minimum_required(VERSION 3.10)

set(LIB_NAME ImGUI)

file(GLOB IMGUI_SRC 
	"${PROJECT_SOURCE_DIR}/external/imgui/imgui.cpp"
	"${PROJECT_SOURCE_DIR}/external/imgui/imgui_demo.cpp"
	"${PROJECT_SOURCE_DIR}/external/imgui/imgui_draw.cpp"
	"${PROJECT_SOURCE_DIR}/external/imgui/imgui_widgets.cpp"
	"${PROJECT_SOURCE_DIR}/external/imgui/imgui_tables.cpp")

file(                                                                           
    GLOB_RECURSE LIB_SRC                                                        #利用glob命令读取扢�有源文件list
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c*"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
)

list(FILTER LIB_SRC EXCLUDE REGEX .*deprecated/.*)                              #排除deprecated 目录下面的所有文�??

add_library(${LIB_NAME} STATIC ${LIB_SRC} ${IMGUI_SRC})

add_compile_definitions(_ENABLE_EXTENDED_ALIGNED_STORAGE)

foreach(SRC IN ITEMS ${LIB_SRC})                                                #为VS工程添加filter 方便查看文件结构目录
    get_filename_component(SRC_PATH "${SRC}" PATH)
    file(RELATIVE_PATH SRC_PATH_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${SRC_PATH}")
    string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
    source_group("${GROUP_PATH}" FILES "${SRC}")
endforeach()

source_group("external" FILES ${IMGUI_SRC})

#if(WIN32)
#    target_compile_options(${LIB_NAME} PRIVATE -Xcompiler "/wd 4819")               #禁止编译时报告文件编码不是unicode的warning，由于cuda头文件都不是unicode。使编译报错更清�??
#endif()
file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                 #判断当前project在根目录下的相对路径
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Rendering/GUI")      #为project设定folder目录
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)    #cmake 会自动延迟cuda dlink操作直到静��库被编译成动��库或��exe�??
                                                                                #但这样影响了后面其他工程对PhysLeo.lib的引用，现在还有些原因未明，暂时在编译成库之前开启dlink
set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "px${LIB_NAME}-${PHOENIX_PHYSICS_VERSION}")

target_include_directories(${LIB_NAME} PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/Rendering/GUI>)

target_link_libraries(${LIB_NAME} Core IO Framework VkRenderEngine)
