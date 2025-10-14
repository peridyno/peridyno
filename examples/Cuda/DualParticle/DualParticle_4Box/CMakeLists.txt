if(PERIDYNO_LIBRARY_DUALPARTICLESYSTEM)
    set(LIB_DEPENDENCY     
        DualParticleSystem 
        Multiphysics
        UbiGUI)
    add_example(DualParticle_4Box DualParticle LIB_DEPENDENCY)
endif()