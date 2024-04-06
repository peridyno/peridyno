if(PERIDYNO_LIBRARY_DUALPARTICLESYSTEM)
    set(LIB_DEPENDENCY     
        Core 
        Framework 
        IO          
        QtGUI
        DualParticleSystem 
        GlfwGUI
        GLRenderEngine)
    add_example(DualParticle_4Box DualParticle LIB_DEPENDENCY)
endif()