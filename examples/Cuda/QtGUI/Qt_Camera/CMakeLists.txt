set(LIB_DEPENDENCY 
    Modeling
    QtGUI)

if(MSVC)
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

add_example(Qt_Camera Qt LIB_DEPENDENCY)