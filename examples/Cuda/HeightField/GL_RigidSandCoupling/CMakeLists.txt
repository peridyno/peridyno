if(PERIDYNO_LIBRARY_HEIGHTFIELD)
    set(LIB_DEPENDENCY 
        Modeling
        RigidBody
        HeightField
        ParticleSystem
        Peridynamics
        DualParticleSystem
        SemiAnalyticalScheme
        Multiphysics
        UbiGUI
        )
    add_example(GL_RigidSandCoupling HeightField LIB_DEPENDENCY)
endif()