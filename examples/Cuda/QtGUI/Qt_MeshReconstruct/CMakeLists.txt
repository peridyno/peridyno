if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_DUALPARTICLESYSTEM)
    set(LIB_DEPENDENCY     
        Core 
        Framework 
        IO 
        DualParticleSystem 
        RigidBody
        ParticleSystem  
        QtGUI
        GLRenderEngine)
    add_example(Qt_MeshReconstruct Qt LIB_DEPENDENCY)
endif()

