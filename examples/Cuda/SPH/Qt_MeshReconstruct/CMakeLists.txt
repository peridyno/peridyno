if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_DUALPARTICLESYSTEM)
    set(LIB_DEPENDENCY     
        ParticleSystem 
        Multiphysics
        QtGUI)
    add_example(Qt_MeshReconstruct SPH LIB_DEPENDENCY)
endif()

