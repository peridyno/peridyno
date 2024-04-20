if(PERIDYNO_LIBRARY_DUALPARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY     
        Core 
        Framework 
        IO          
        QtGUI
        DualParticleSystem 
        SemiAnalyticalScheme
        ParticleSystem
        GLRenderEngine)
    add_example(Qt_DualParticle_FishAndBall DualParticle LIB_DEPENDENCY)
endif()