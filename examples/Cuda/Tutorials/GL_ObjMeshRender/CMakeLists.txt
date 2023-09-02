if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY 
        Core 
        Framework
        QtGUI
        GLRenderEngine)

    add_example(GL_ObjMeshRender Tutorials LIB_DEPENDENCY)
endif()
