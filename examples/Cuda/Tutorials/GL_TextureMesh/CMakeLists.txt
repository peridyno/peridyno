if(PERIDYNO_PLUGIN_OBJ)
	set(LIB_DEPENDENCY 
		Core 
		Framework
		ObjIO
		GlfwGUI
		GLRenderEngine)

	add_example(GL_TextureMesh Tutorials LIB_DEPENDENCY)
endif()
