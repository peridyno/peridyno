if(PERIDYNO_LIBRARY_DUALPARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY     
        Core 
        Framework 
        UbiGUI
        DualParticleSystem 
        SemiAnalyticalScheme
        ParticleSystem
        GLRenderEngine)
    add_example(DualParticle_KissingFish DualParticle LIB_DEPENDENCY)
endif()