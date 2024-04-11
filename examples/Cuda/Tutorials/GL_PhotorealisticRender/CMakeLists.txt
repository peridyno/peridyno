if(PERIDYNO_PLUGIN_OBJ)
	set(LIB_DEPENDENCY 
		Core 
		Framework
		ObjIO
		GlfwGUI
		GLRenderEngine)

	add_example(GL_PhotorealisticRender Tutorials LIB_DEPENDENCY)
endif()
