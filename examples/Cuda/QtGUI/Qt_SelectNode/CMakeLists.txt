#if(PERIDYNO_Qt5_GUI)
    set(LIB_DEPENDENCY 
        Core 
        Framework 
        Modeling 
        QtGUI)
    add_example(Qt_SelectNode Qt LIB_DEPENDENCY)
#endif