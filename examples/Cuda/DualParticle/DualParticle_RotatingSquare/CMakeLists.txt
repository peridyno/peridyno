if(PERIDYNO_LIBRARY_DUALPARTICLESYSTEM)
    set(LIB_DEPENDENCY     
        DualParticleSystem 
        Multiphysics
        GlfwGUI
        )
    add_example(DualParticle_RotatingSquare DualParticle LIB_DEPENDENCY)
endif()