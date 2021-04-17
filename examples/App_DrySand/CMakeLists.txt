set(PROJECT_NAME App_DrySand)

link_libraries(Core Framework IO Graphics)
link_libraries(ParticleSystem RigidBody)
link_libraries(GlfwGUI)

set(SRC_DIR "${PROJECT_SOURCE_DIR}/examples/${PROJECT_NAME}")

file(                                                                                                       #利用glob命令读取所有源文件list
    GLOB_RECURSE SRC_LIST 
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${SRC_DIR}/*.c*"
    "${SRC_DIR}/*.h*"
)

add_executable(${PROJECT_NAME} ${SRC_LIST})                                                                 #添加编译目标 可执行文件

target_link_libraries(${PROJECT_NAME} 
	Core 
	Framework 
	IO 
	Graphics 
	ParticleSystem 
	RigidBody 
	GlfwGUI)

file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                  #判断当前project在根目录下的相对路径
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Examples")                              #为project设定folder目录
#    set(EXECUTABLE_OUTPUT_PATH  ${CMAKE_CURRENT_BINARY_DIR}/bin/)

if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
elseif(UNIX)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
    else()
        set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
    endif()
endif()   

foreach(SRC IN ITEMS ${SRC_LIST})                                                                           #为VS工程添加filter 方便查看文件结构目录
    get_filename_component(SRC_PATH "${SRC}" PATH)
    file(RELATIVE_PATH SRC_PATH_REL "${SRC_DIR}" "${SRC_PATH}")
    string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
    source_group("${GROUP_PATH}" FILES "${SRC}")
endforeach()