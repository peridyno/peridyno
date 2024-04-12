if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        GlfwGUI)
    add_example(GL_PhotorealisticInstanceRender Tutorials LIB_DEPENDENCY)
endif()