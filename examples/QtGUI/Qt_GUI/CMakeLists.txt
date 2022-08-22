cmake_minimum_required(VERSION 3.10)

if(PERIDYNO_Qt5_GUI)

    set(PROJECT_NAME Qt_GUI)

    link_directories("${PROJECT_SOURCE_DIR}/Source")

    if(Qt5_FOUND)
        message("Find QT5")
        link_libraries(Qt5::Widgets Qt5::Core Qt5::Gui)
    endif()

    file(
        GLOB_RECURSE SRC_LIST 
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.c*"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
    )
	source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_LIST})

    add_executable(${PROJECT_NAME} ${SRC_LIST})
    
    if(UNIX)
		#the option "-Wl,--whole-archive" is an equivalance to /WHOLEARCHIVE
        target_link_libraries(${PROJECT_NAME} PUBLIC   
            -Wl,--whole-archive
            Core 
            Framework 
            Modeling
            Interaction
            IO 
            ParticleSystem  
            QtGUI
			RigidBody
            ImWidgets
			GLRenderEngine
            -Wl,--no-whole-archive) 

    elseif(WIN32)
        target_link_libraries(${PROJECT_NAME} PUBLIC
            Core 
            Framework 
            Modeling
            Interaction
            IO 
            ParticleSystem 
			RigidBody
            QtGUI
            ImWidgets
			GLRenderEngine) 

		#/WHOLEARCHIVE is used to link unused static variables for visual studio, which is necessary for auto class registration.
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_DEBUG           
           "/WHOLEARCHIVE:dynoFramework-${PERIDYNO_LIBRARY_VERSION}d.lib
            /WHOLEARCHIVE:dynoModeling-${PERIDYNO_LIBRARY_VERSION}d.lib
            /WHOLEARCHIVE:dynoInteraction-${PERIDYNO_LIBRARY_VERSION}d.lib
            /WHOLEARCHIVE:dynoParticleSystem-${PERIDYNO_LIBRARY_VERSION}d.lib
            /WHOLEARCHIVE:dynoIO-${PERIDYNO_LIBRARY_VERSION}d.lib
            /WHOLEARCHIVE:dynoQtGUI-${PERIDYNO_LIBRARY_VERSION}d.lib
            /WHOLEARCHIVE:GLRenderEngined.lib
            /WHOLEARCHIVE:ImWidgetsd.lib
            /WHOLEARCHIVE:RenderCored.lib")

        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_RELEASE 
           "/WHOLEARCHIVE:dynoFramework-${PERIDYNO_LIBRARY_VERSION}.lib
            /WHOLEARCHIVE:dynoModeling-${PERIDYNO_LIBRARY_VERSION}.lib
            /WHOLEARCHIVE:dynoInteraction-${PERIDYNO_LIBRARY_VERSION}.lib
            /WHOLEARCHIVE:dynoParticleSystem-${PERIDYNO_LIBRARY_VERSION}.lib
            /WHOLEARCHIVE:dynoIO-${PERIDYNO_LIBRARY_VERSION}.lib
            /WHOLEARCHIVE:dynoQtGUI-${PERIDYNO_LIBRARY_VERSION}.lib
            /WHOLEARCHIVE:GLRenderEngine.lib
            /WHOLEARCHIVE:ImWidgets.lib
            /WHOLEARCHIVE:RenderCore.lib")
    else()
        message(FATAL_ERROR "Could not set install folders for this platform!")
    endif()

    file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Examples/Qt")
    set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

