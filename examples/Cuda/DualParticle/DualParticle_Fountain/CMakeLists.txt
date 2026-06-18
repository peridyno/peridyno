if(PERIDYNO_LIBRARY_DUALPARTICLESYSTEM)
    set(LIB_DEPENDENCY     
        Core 
        Framework 
        IO          
        ParticleSystem 
        UbiGUI
        GLRenderEngine)
    add_example(DualParticle_Fountain DualParticle LIB_DEPENDENCY)
endif()