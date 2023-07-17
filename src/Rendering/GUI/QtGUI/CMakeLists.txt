cmake_minimum_required(VERSION 3.10)

# Macro definition for ToolBar
add_definitions(-DTT_BUILDING_DLL)

set(LIB_NAME QtGUI)
set(LIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

#set(CMAKE_AUTOMOC ON)                                                                   # 弢�启自动moc功能
#set(CMAKE_AUTOUIC ON) 

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)

if (${QT_VERSION_MAJOR} EQUAL 6)
    find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS 
        Core
        Core5Compat
        Widgets
        Gui
        OpenGL
        OpenGLWidgets
        Svg)

    qt_add_resources(RESOURCES 
        ./Resources/DefaultStyle.qrc
        ./Resources/StyleTemplate.qrc
        ./Resources/img/icons.qrc)
else()
    find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS 
        Core
        Widgets
        Gui
        OpenGL
        Svg)

    qt5_add_resources(RESOURCES 
        ./Resources/DefaultStyle.qrc
        ./Resources/StyleTemplate.qrc
        ./Resources/img/icons.qrc)
endif()

file(                                                                           
    GLOB_RECURSE ALL_FILES                                                        #利用glob命令读取扢�有源文件list
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${LIB_SRC_DIR}/*.c*"
    "${LIB_SRC_DIR}/*.h*"
)

file(                                                                           
    GLOB_RECURSE MOC_FILES                                                        #利用glob命令读取扢�有源文件list
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${LIB_SRC_DIR}/*.h*"
)

if (${QT_VERSION_MAJOR} EQUAL 6)
  qt_wrap_cpp(
    ALL_FILES
      ${MOC_FILES}
  )
else()
  qt5_wrap_cpp(
    ALL_FILES
      ${MOC_FILES}
  )
endif()


add_library(${LIB_NAME} SHARED 
	${ALL_FILES}
	${RESOURCES})

foreach(SRC IN ITEMS ${ALL_FILES})
    get_filename_component(SRC_PATH "${SRC}" PATH)
    file(RELATIVE_PATH SRC_PATH_REL "${LIB_SRC_DIR}" "${SRC_PATH}")
    string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
    source_group("${GROUP_PATH}" FILES "${SRC}")
endforeach()

foreach(SRC IN ITEMS ${RESOURCES})
    get_filename_component(SRC_PATH "${SRC}" PATH)
    file(RELATIVE_PATH SRC_PATH_REL "${LIB_SRC_DIR}" "${SRC_PATH}")
    string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
    source_group("${GROUP_PATH}" FILES "${SRC}")
endforeach()

if(WIN32)
    target_compile_options(${LIB_NAME} PRIVATE -Xcompiler "/wd 4819")
endif()
file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Rendering/GUI")
if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_FLAGS}")
endif()                                                                      

set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "dyno${LIB_NAME}-${PERIDYNO_LIBRARY_VERSION}")
    
if(WIN32)
    set_target_properties(${LIB_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
elseif(UNIX)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set_target_properties(${LIB_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
    else()
        set_target_properties(${LIB_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
    endif()
endif()

# target_link_libraries(${LIB_NAME} Core Framework ParticleSystem RenderEngine glu32)
target_link_libraries(${LIB_NAME} PUBLIC
	Core Framework RenderCore GLRenderEngine ImWidgets imgui
	Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Core 
    Qt${QT_VERSION_MAJOR}::Gui 
    Qt${QT_VERSION_MAJOR}::OpenGL 
    Qt${QT_VERSION_MAJOR}::Svg)

if (${QT_VERSION_MAJOR} EQUAL 6)
    target_link_libraries(${LIB_NAME} PUBLIC Qt6::Core5Compat Qt6::OpenGLWidgets)  
endif()

target_link_libraries(${LIB_NAME} PRIVATE $<BUILD_INTERFACE:nodes>)  

target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/GUI>
    $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/GUI/QtGUI>	
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/Rendering/GUI>)

install(TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}Targets
    RUNTIME  DESTINATION  ${PERIDYNO_RUNTIME_INSTALL_DIR}
    LIBRARY  DESTINATION  ${PERIDYNO_LIBRARY_INSTALL_DIR}
    ARCHIVE  DESTINATION  ${PERIDYNO_ARCHIVE_INSTALL_DIR}
    )

install(EXPORT ${LIB_NAME}Targets DESTINATION ${PERIDYNO_CMAKE_CONFIG_INSTALL_DIR}
    FILE ${LIB_NAME}Targets.cmake)

#Append ${LIB_NAME}Targets.cmake to the global list, which will be include in PeridynoConfig.cmake
get_property(LOCAL_CMAKES_NAMES GLOBAL PROPERTY "GLOBAL_CMAKES_NAMES")
list(APPEND LOCAL_CMAKES_NAMES "${LIB_NAME}Targets.cmake")    
set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES ${LOCAL_CMAKES_NAMES})

file(GLOB FILE_QTGUI_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${FILE_QTGUI_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/GUI/QtGUI)

file(GLOB FILE_QTGUI_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/ToolBar/*.h")
install(FILES ${FILE_QTGUI_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/GUI/QtGUI/ToolBar)

file(GLOB FILE_QTGUI_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Nodes/*.h")
install(FILES ${FILE_QTGUI_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/GUI/QtGUI/Nodes)

file(GLOB FILE_QTGUI_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Resources/*.*")
install(FILES ${FILE_QTGUI_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/GUI/QtGUI/Resources)
