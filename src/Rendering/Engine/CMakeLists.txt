# internal OpenGL rendering support
add_subdirectory(OpenGL)

if(PERIDYNO_PLUGIN_VTK)
	find_package(VTK 9.0)

	if(VTK_FOUND)
		add_subdirectory(VTK)
	endif()
endif()