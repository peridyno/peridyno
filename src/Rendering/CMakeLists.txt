add_subdirectory(Core)
add_subdirectory(Engine)

if(PERIDYNO_LIBRARY_FRAMEWORK AND PERIDYNO_LIBRARY_GUI)
    add_subdirectory(GUI)
endif()