if(PERIDYNO_LIBRARY_DUALPARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY     
        Core 
        Framework 
        IO          
        QtGUI
        DualParticleSystem 
        SemiAnalyticalScheme
        ParticleSystem
        Multiphysics
        GLRenderEngine)
    add_example(Qt_MPM_Dambreak DualParticle LIB_DEPENDENCY)
endif()