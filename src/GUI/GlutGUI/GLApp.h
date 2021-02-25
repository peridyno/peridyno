/*
 * @file glut_window.h 
 * @Brief Glut-based window.
 * @author Fei Zhu
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */
#pragma once
#include <glm/vec4.hpp>
#include "../AppBase.h"
#include "Camera.h"

namespace dyno {

typedef glm::vec4 Color;

class GLApp : public AppBase
{
public:
    GLApp();
   ~GLApp();

    void createWindow(int width, int height) override; //create window with the parameters set
    void closeWindow();  //close window
    const std::string& name() const;
    int getWidth() const;
    int getHeight() const;
	void setWidth(int width);
	void setHeight(int height);

    void enableEventMode();
    void disableEventMode();

    //save screenshot to file
    bool saveScreen(const std::string &file_name) const;  //save to file with given name
    bool saveScreen();                                    //save to file with default name "screen_capture_XXX.png"
	
	void enableSaveScreen() { m_bSaveScreen = true; }
	void disableSaveScreen() { m_bSaveScreen = false; };
	void setOutputPath(std::string path) { m_outputPath = path; }
	void setSaveScreenInterval(int n) { m_saveScreenInterval = n < 1 ? 1 : n; }
	int getSaveScreenInternal() { return m_saveScreenInterval; }

	bool isActive() { return m_bAnimate; }
	bool isSaveScreen() { return m_bSaveScreen; }
	void drawString(std::string s, const Color &color, int x, int y);

    //display frame-rate
    void drawFrameRate();  
    void enableDisplayFrameRate();
    void disableDisplayFrameRate();
	bool isShowFrameRate();
	bool isShowBoundingBox();

	void enableBackground();
	void disableBackground();
	void enableSceneBoundary();
	void disableSceneBoundary();
	bool isShowBackground();

    //advanced: 
    //set custom callback functions
    void setDisplayFunction(void (*func)(void));  
    void setIdleFunction(void (*func)(void));  
    void setReshapeFunction(void (*func)(int width, int height));
    void setKeyboardFunction(void (*func)(unsigned char key, int x, int y));
    void setSpecialFunction(void (*func)(int key, int x, int y));
    void setMotionFunction(void (*func)(int x, int y));
    void setMouseFunction(void (*func)(int button, int state, int x, int y));
    void setMouseWheelFunction(void(*func)(int wheel, int direction, int x, int y));
    void setInitFunction(void (*func)(void)); //the init function before entering mainloop

	void setButtonType(int button) { m_buttonType = button; }
	void setButtonState(int status) { m_buttonStatus = status; }

	int getButtonType() { return m_buttonType; }
	int getButtonStatus() { return m_buttonStatus; }

	Camera& activeCamera() { return m_camera; }

    static void bindDefaultKeys(unsigned char key, int x, int y);  //bind the default keyboard behaviors
    
	void mainLoop() override;
	void setSecondaryLineNumber(int num);

protected:
    //default callback functions
    static void displayFunction(void);                                       //display all render tasks provided by user
    static void idleFunction(void);                                          //do nothing
    static void reshapeFunction(int width, int height);                      //adjust view port to reveal the change
    static void keyboardFunction(unsigned char key, int x, int y);           //press 'ESC' to close window, ect.
    static void specialFunction(int key, int x, int y);                      //do nothing
    static void motionFunction(int x, int y);                                //left button: rotate, middle button: zoom, right button: translate
    static void mouseFunction(int button, int state, int x, int y);          //keep track of mouse state
    static void mouseWheelFunction(int wheel, int direction, int x, int y);  //mouse wheel: zoom
    static void initFunction(void);                                          //init viewport and background color

    void initOpenGLContext();
    void initCallbacks();    //init default callbacks
    void initDefaultLight(); //init a default light

protected:
    //pointers to callback methods
    void(*display_function_)(void);
    void(*idle_function_)(void);
    void(*reshape_function_)(int width, int height);
    void(*keyboard_function_)(unsigned char key, int x, int y);
    void(*special_function_)(int key, int x, int y);
    void(*motion_function_)(int x, int y);
    void(*mouse_function_)(int button, int state, int x, int y);
    void(*mouse_wheel_function_)(int wheel, int direction, int x, int y);
    void(*init_function_)(void);

	void drawBackground();
	void drawAxis();
	void drawBoundingBox(Vector3f lo, Vector3f hi);

protected:
    //basic information of window
    std::string m_winName;
    int m_winID;

    unsigned int m_width;
    unsigned int m_height;

	int m_secLineNum;

	Color background_color_; //use double type in order not to make GlutWindow template
	Color text_color_;       //the color to display text, e.g. fps

    //state of the mouse
	int m_buttonType;
	int m_buttonStatus;

    //fps display
    bool display_fps_;

    //event mode
    bool event_mode_;

	bool m_bAnimate;
	bool m_bSaveScreen = false;
	bool m_bShowBackground;
	bool m_bShowBoundingbox = false;

	int m_saveScreenInterval = 1;
    
	std::string m_outputPath;

    //current screen capture file index
    unsigned int screen_capture_file_index_;

	Camera m_camera;
};
}  //end of namespace dyno
