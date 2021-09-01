# internal OpenGL rendering support
add_subdirectory(OpenGL)

# VTK rendering support
option(PERIDYNO_PLUGIN_VTK "Enable VTK plugin for visualization" ON)

if(PERIDYNO_PLUGIN_VTK)
	find_package(VTK 9.0)
	if(VTK_FOUND)
		add_subdirectory(VTK)
	endif()
endif()