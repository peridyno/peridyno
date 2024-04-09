if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        GlfwGUI)
    add_example(GL_InstancePhotorealisticRender Tutorials LIB_DEPENDENCY)
endif()