set(LIB_DEPENDENCY 
    Core 
    Framework
    GlfwGUI
    GLRenderEngine)

add_example(GL_ObjMeshRender Tutorials LIB_DEPENDENCY)

file(COPY  "data"   DESTINATION ${CMAKE_CURRENT_BINARY_DIR})