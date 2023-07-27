set(LIB_DEPENDENCY 
    QtGUI)

if(MSVC)
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

add_example(Qt_ShowChinese Qt LIB_DEPENDENCY)