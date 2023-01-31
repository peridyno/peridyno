if(PERIDYNO_LIBRARY_PERIDYNAMICS AND PERIDYNO_PLUGIN_VTK)
    set(PROJECT_NAME Vtk_Elasticity)

    file(                                                                                                       #利用glob命令读取所有源文件list
        GLOB_RECURSE SRC_LIST 
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.c*"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
    )

    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_LIST})

    add_executable(${PROJECT_NAME} ${SRC_LIST})                                                                 #添加编译目标 可执行文件

    target_link_libraries(${PROJECT_NAME} 
        Core 
        Framework 
        IO 
        Peridynamics 
        RigidBody 
        GlfwGUI
		VtkRenderEngine)

    file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                  #判断当前project在根目录下的相对路径
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Examples/VTK")                              #为project设定folder目录
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
endif()