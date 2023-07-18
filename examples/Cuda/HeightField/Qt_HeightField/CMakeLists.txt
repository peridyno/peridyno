if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_HEIGHTFIELD)
    set(LIB_DEPENDENCY 
        HeightField
        QtGUI)
    add_example(Qt_HeightField HeightField LIB_DEPENDENCY)
endif()

