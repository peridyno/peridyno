cmake_minimum_required(VERSION 3.10)

add_definitions(-DTT_BUILDING_DLL)

set(LIB_NAME QtGUI)
set(LIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_AUTOMOC ON)                                                                   # 开启自动moc功能
set(CMAKE_AUTOUIC ON) 

find_package(Qt5 COMPONENTS
             Core
             Widgets
             Gui
             OpenGL
			 Svg)

qt5_add_resources(RESOURCES 
    ./Resources/resources.qrc
    ./Resources/StyleTemplate.qrc)


file(                                                                           
    GLOB_RECURSE LIB_SRC                                                        #利用glob命令读取所有源文件list
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${LIB_SRC_DIR}/*.c*"
    "${LIB_SRC_DIR}/*.h*"
	"${LIB_SRC_DIR}/Nodes/*.h*"
    "${LIB_SRC_DIR}/Nodes/*.c*"
    "${LIB_SRC_DIR}/ToolBar/*.h*"
	"${LIB_SRC_DIR}/ToolBar/*.c*"
)

add_library(${LIB_NAME} STATIC 
	${LIB_SRC} 
	${RESOURCES})

foreach(SRC IN ITEMS ${LIB_SRC})
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
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
                                                                             

set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "dyno${LIB_NAME}-${PERIDYNO_LIBRARY_VERSION}")
    
# target_link_libraries(${LIB_NAME} Core Framework ParticleSystem RenderEngine glu32)
target_link_libraries(${LIB_NAME} PUBLIC
	Core Framework RenderCore GLRenderEngine ImWidgets imgui nodes
	Qt5::Widgets Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::Svg)  

target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>	
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
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
