if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(PROJECT_NAME GL_Timing)
    
    file(
        GLOB_RECURSE SRC_LIST 
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.c*"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
    )

	source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_LIST})

    add_executable(${PROJECT_NAME} ${SRC_LIST})
	target_link_libraries(${PROJECT_NAME} 
        Core 
        Framework 
        IO
        RigidBody 
        GlfwGUI
        GLRenderEngine)

    file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Examples/Tutorials")

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