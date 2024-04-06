if(PERIDYNO_LIBRARY_DUALPARTICLESYSTEM)
    set(LIB_DEPENDENCY     
        Core 
        Framework 
        IO 
        DualParticleSystem 
        GlfwGUI
        QtGUI
        GLRenderEngine)
    add_example(DualParticle_RotatingSquare DualParticle LIB_DEPENDENCY)
endif()