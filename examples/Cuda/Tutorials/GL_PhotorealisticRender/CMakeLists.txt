if(PERIDYNO_PLUGIN_OBJ)
	set(LIB_DEPENDENCY 
        Modeling
        RigidBody
        ParticleSystem
        Peridynamics
        Multiphysics
        UbiGUI
        ObjIO
		GLRenderEngine)

	add_example(GL_PhotorealisticRender Tutorials LIB_DEPENDENCY)
endif()
