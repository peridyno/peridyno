if(PERIDYNO_LIBRARY_DUALPARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY     
        Core 
        Framework 
        IO          
        GlfwGUI
        DualParticleSystem 
        GLRenderEngine)
    add_example(DualParticle_CollidingFish DualParticle LIB_DEPENDENCY)
endif()