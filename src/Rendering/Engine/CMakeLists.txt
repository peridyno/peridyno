# internal OpenGL rendering support
if("${PERIDYNO_GPU_BACKEND}" STREQUAL "Vulkan")
    add_subdirectory(VkRenderEngine)
endif()

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
    add_subdirectory(OpenGL)

	if(PERIDYNO_PLUGIN_VTK)
		find_package(VTK 9.0)

		if(VTK_FOUND)
			add_subdirectory(VTK)
		endif()
	endif()

endif()