/*
 * @file glut_window.cpp 
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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Utility.h"
#include "Image_IO/image_io.h"
#include "GLApp.h"
#include "OpenGLContext.h"

namespace dyno {

GLApp::GLApp()
    :m_winName(std::string("Peridyno 1.0")),m_winID(-1),m_width(640),m_height(480),
     display_fps_(true),screen_capture_file_index_(0),event_mode_(false)
	, m_bAnimate(false)
	, m_secLineNum(10)
	, m_bShowBackground(true)
{
    background_color_ = Color(0.6, 0.6, 0.6, 1.0);
    text_color_ = Color(1.0f, 1.0f, 1.0f, 1.0f);
}

GLApp::~GLApp()
{
}

void GLApp::createWindow(int width, int height)
{
	m_width = width;
	m_height = height;
	initCallbacks();
	
	int argc = 1;
	const int max_length = 1024; //assume length of the window name does not exceed 1024 characters
	char *argv[1];
	char name_str[max_length];
	std::string win_title = std::string("Peridyno ") + std::to_string(PERIDYNO_VERSION_MAJOR) + std::string(".") + std::to_string(PERIDYNO_VERSION_MINOR) + std::string(".") + std::to_string(PERIDYNO_VERSION_PATCH);
	strcpy(name_str, win_title.c_str());
	argv[0] = name_str;

	glutInit(&argc, argv);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);  //this option allows leaving the glut loop without exit program
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_ALPHA);
	glutInitWindowSize(m_width, m_height);
	m_winID = glutCreateWindow(win_title.c_str());
	glutHideWindow();

    glutShowWindow();
    glutSetWindowData(this);  //bind 'this' pointer with the window
    glutDisplayFunc(display_function_);
    glutIdleFunc(idle_function_);
    glutReshapeFunc(reshape_function_);
    glutKeyboardFunc(keyboard_function_);
    glutSpecialFunc(special_function_);
    glutMotionFunc(motion_function_);
    glutMouseFunc(mouse_function_);
    glutMouseWheelFunc(mouse_wheel_function_);

    (*init_function_)(); //call the init function before entering main loop

	m_camera.registerPoint(0.5f, 0.5f);
	m_camera.translateToPoint(0, 0);

	m_camera.zoom(3.0f);
	m_camera.setGL(0.01f, 3.0f, (float)getWidth(), (float)getHeight());
}

void GLApp::closeWindow()
{
    glutLeaveMainLoop();
}

const std::string& GLApp::name() const
{
    return m_winName;
}

int GLApp::getWidth() const
{
    if(glutGet(GLUT_INIT_STATE))  //window is created
        return glutGet(GLUT_WINDOW_WIDTH);
    else
        return m_width;
}

int GLApp::getHeight() const
{
    if(glutGet(GLUT_INIT_STATE)) //window is created
        return glutGet(GLUT_WINDOW_HEIGHT);
    else
        return m_height;
}

void GLApp::setWidth(int width)
{
	m_width = width;
}

void GLApp::setHeight(int height)
{
	m_height = height;
}

void GLApp::enableEventMode()
{
    this->event_mode_ = true;
}

void GLApp::disableEventMode()
{
    this->event_mode_ = false;
}

////////////////////////////////////////////////// screen shot and display frame-rate////////////////////////////////////////////////////////////////

bool GLApp::saveScreen(const std::string &file_name) const
{
    int width = this->getWidth(), height = this->getHeight();
    unsigned char *data = new unsigned char[width*height*3];  //RGB
    assert(data);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,(void*)data);
    Image image(width,height,Image::RGB,data);
    image.flipVertically();
    bool status = ImageIO::save(file_name,&image);
    delete[] data;
    return status;
}

bool GLApp::saveScreen()
{
	std::stringstream adaptor;
	adaptor << screen_capture_file_index_++;
	std::string index_str;
	adaptor >> index_str;
	std::string file_name = m_outputPath + std::string("screen_capture_") + index_str + std::string(".png");
	return saveScreen(file_name);
}

void GLApp::drawFrameRate()
{
	if (!glutGet(GLUT_INIT_STATE))  //window is not created
		exit(0);
    if(display_fps_)
    {
        static unsigned int frame = 0, time = 0, time_base = 0;
        double fps = 60.0;
        ++frame;
        time = glutGet(GLUT_ELAPSED_TIME); //millisecond
        if(time - time_base > 10) // compute every 10 milliseconds
        {
            fps = frame*1000.0/(time-time_base);
            time_base = time;
            frame = 0;
        }
        std::stringstream adaptor;
        adaptor.precision(2);
        std::string str;
        if(fps>1.0)  //show fps
        {
            adaptor<<fps;
            str = std::string(" Frame rate: ") + adaptor.str();
        }
        else  //show spf
        {
            assert(fps>0);
            adaptor<< 1.0/fps;
            str = std::string("SPF: ") + adaptor.str();
        }
       
		drawString(str, Color(1.0f, 1.0f, 1.0f, 1.0f), 0, getHeight()-25);
    }
}

void GLApp::enableDisplayFrameRate()
{
    display_fps_ = true;
}

void GLApp::disableDisplayFrameRate()
{
    display_fps_ = false;
}

bool GLApp::isShowFrameRate()
{
	return display_fps_;
}

bool GLApp::isShowBoundingBox()
{
	return m_bShowBoundingbox;
}

void GLApp::enableBackground()
{
	m_bShowBackground = true;
}

void GLApp::disableBackground()
{
	m_bShowBackground = false;
}

void GLApp::enableSceneBoundary()
{
	m_bShowBoundingbox = true;
}

void GLApp::disableSceneBoundary()
{
	m_bShowBoundingbox = false;
}

bool GLApp::isShowBackground()
{
	return m_bShowBackground;
}

////////////////////////////////////////////////// set custom callback functions ////////////////////////////////////////////////////////////////////

void GLApp::setDisplayFunction(void (*func)(void))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        display_function_ = GLApp::displayFunction;
    }
    else
        display_function_ = func;
}

void GLApp::setIdleFunction(void (*func)(void))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        idle_function_ = GLApp::idleFunction;
    }
    else
        idle_function_ = func;
}

void GLApp::setReshapeFunction(void (*func)(int width, int height))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        reshape_function_ = GLApp::reshapeFunction;
    }
    else
        reshape_function_ = func;
}

void GLApp::setKeyboardFunction(void (*func)(unsigned char key, int x, int y))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        keyboard_function_ = GLApp::keyboardFunction;
    }
    else
        keyboard_function_ = func;
}

void GLApp::setSpecialFunction(void (*func)(int key, int x, int y))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        special_function_ = GLApp::specialFunction;
    }
    else
        special_function_ = func;
}

void GLApp::setMotionFunction(void (*func)(int x, int y))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        motion_function_ = GLApp::motionFunction;
    }
    else
        motion_function_ = func;
}

void GLApp::setMouseFunction(void (*func)(int button, int state, int x, int y))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        mouse_function_ = GLApp::mouseFunction;
    }
    else
        mouse_function_ = func;
}

void GLApp::setMouseWheelFunction(void(*func)(int wheel, int direction, int x, int y))
{
    if (func == NULL)
    {
        std::cerr << "NULL callback function provided, use default instead.\n";
        mouse_wheel_function_ = GLApp::mouseWheelFunction;
    }
    else
        mouse_wheel_function_ = func;
}

void GLApp::setInitFunction(void (*func)(void))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        init_function_ = GLApp::initFunction;
    }
    else
        init_function_ = func;
}

void GLApp::bindDefaultKeys(unsigned char key, int x, int y)
{
    GLApp::keyboardFunction(key,x,y);
}

void GLApp::mainLoop()
{
	OpenGLContext::getInstance().initialize();
	SceneGraph::getInstance().initialize();

	if (event_mode_ == false)
		glutMainLoop();
}

void GLApp::setSecondaryLineNumber(int num)
{
	m_secLineNum = num;
}

////////////////////////////////////////////////// default callback functions ////////////////////////////////////////////////////////////////////

void GLApp::displayFunction(void)
{
    GLApp * cur_window = (GLApp*)glutGetWindowData();
	SceneGraph& scenegraph = SceneGraph::getInstance();

    Color background_color = cur_window->background_color_;

    glClearColor(background_color.r, background_color.g, background_color.b, background_color.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();

	glPushMatrix();

	glPushMatrix();

	glPushMatrix();

	if (cur_window->isShowBackground())
	{
		cur_window->drawBackground();
	}
	
	if (cur_window->isShowFrameRate())
	{
		cur_window->drawFrameRate();
	}

	if (cur_window->isShowBoundingBox())
	{
		cur_window->drawBoundingBox(scenegraph.getLowerBound(), scenegraph.getUpperBound());
	}

	scenegraph.draw();

	glPopMatrix();

    glutPostRedisplay();
    glutSwapBuffers();
}

void GLApp::idleFunction(void)
{
	SceneGraph& scenegraph = SceneGraph::getInstance();

	GLApp * cur_window = (GLApp*)glutGetWindowData();
	if(cur_window->isActive())
	{ 
		scenegraph.takeOneFrame();
		if (cur_window->isSaveScreen() && scenegraph.getFrameNumber() % cur_window->getSaveScreenInternal() == 0)
		{
			cur_window->saveScreen();
		}
	}
	
    glutPostRedisplay();
}

void GLApp::reshapeFunction(int width, int height)
{
	GLApp *window = static_cast<GLApp*>(glutGetWindowData());

	glViewport(0, 0, width, height);

 	window->activeCamera().setGL(0.01f, 10.0f, (float)width, (float)height);
 	window->setWidth(width);
 	window->setHeight(height);

	glutPostRedisplay();
}

void GLApp::keyboardFunction(unsigned char key, int x, int y)
{
    GLApp *window = static_cast<GLApp*>(glutGetWindowData());
    assert(window);
    switch(key)
    {
    case 27: //ESC: close window
        glutLeaveMainLoop();
        break;
    case 's': //s: save screen shot
        window->saveScreen();
        break;
    case 'f': //f: enable/disable FPS display
        (window->display_fps_) = !(window->display_fps_);
        break;
	case ' ':
		window->m_bAnimate = !(window->m_bAnimate);
		break;
	case 'j':
		SceneGraph::getInstance().takeOneFrame();
		break;
    default:
        break;
    }
}

void GLApp::specialFunction(int key, int x, int y)
{
}

void GLApp::motionFunction(int x, int y)
{
	GLApp *window = static_cast<GLApp*>(glutGetWindowData());
	Camera& activeCamera = window->activeCamera();

	if (window->getButtonType() == GLUT_LEFT_BUTTON) {
		activeCamera.rotateToPoint(float(x) / float(window->getWidth()) - 0.5f, float(window->getHeight() - y) / float(window->getHeight()) - 0.5f);
	}
	else if (window->getButtonType() == GLUT_RIGHT_BUTTON) {
		activeCamera.translateToPoint(float(x) / float(window->getWidth()) - 0.5f, float(window->getHeight() - y) / float(window->getHeight()) - 0.5f);
	}
	else if (window->getButtonType() == GLUT_MIDDLE_BUTTON) {
		activeCamera.translateLightToPoint(float(x) / float(window->getWidth()) - 0.5f, float(window->getHeight() - y) / float(window->getHeight()) - 0.5f);
	}
	activeCamera.setGL(0.01f, 10.0f, (float)window->getWidth(), (float)window->getHeight());
	glutPostRedisplay();
}

void GLApp::mouseFunction(int button, int state, int x, int y)
{
     GLApp *window = static_cast<GLApp*>(glutGetWindowData());
	 Camera& activeCamera = window->activeCamera();
	 window->setButtonType(button);
	 window->setButtonState(state);

	if (state == GLUT_DOWN) {
		activeCamera.registerPoint(float(x) / float(window->getWidth()) - 0.5f, float(window->getHeight() - y) / float(window->getHeight()) - 0.5f);
	}
}

void GLApp::mouseWheelFunction(int wheel, int direction, int x, int y)
{
	GLApp *window = static_cast<GLApp*>(glutGetWindowData());
	Camera& activeCamera = window->activeCamera();

	switch (direction)
	{
	case 1:
		activeCamera.zoom(-0.3f);
		activeCamera.setGL(0.01f, 10.0f, (float)window->getWidth(), (float)window->getHeight());
		break;
	case -1:
		activeCamera.zoom(0.3f);
		activeCamera.setGL(0.01f, 10.0f, (float)window->getWidth(), (float)window->getHeight());
	default:
		break;
	}
}

void GLApp::initFunction(void)
{
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    GLApp *window = static_cast<GLApp*>(glutGetWindowData());
    assert(window);

    glViewport(0, 0, width, height);        									// set the viewport
    window->initDefaultLight();

    glShadeModel( GL_SMOOTH );
    glClearDepth( 1.0 );														// specify the clear value for the depth buffer
    glEnable( GL_DEPTH_TEST );
    
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );						// specify implementation-specific hints
    Color background_color = window->background_color_;
    
	glClearColor(background_color.r, background_color.g, background_color.b, background_color.a);
}

void GLApp::initOpenGLContext()
{
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		std::cerr << "error: can't init glew!\n";
		std::exit(EXIT_FAILURE);
	}

    std::cout << "openGL Version: " << glGetString(GL_VERSION) << std::endl;

}

void GLApp::initCallbacks()
{
    //set callbacks to default callback functions
    display_function_ = GLApp::displayFunction;
    idle_function_ = GLApp::idleFunction;
    reshape_function_ = GLApp::reshapeFunction;
    keyboard_function_ = GLApp::keyboardFunction;
    special_function_ = GLApp::specialFunction;
    motion_function_ = GLApp::motionFunction;
    mouse_function_ = GLApp::mouseFunction;
    mouse_wheel_function_ = GLApp::mouseWheelFunction;
    init_function_ = GLApp::initFunction;
}

void GLApp::initDefaultLight()
{
    std::cout << "info: add default flash light!" << std::endl;

//     std::shared_ptr<FlashLight> flash_light = std::make_shared<FlashLight>();
//     flash_light->setAmbient(Color4f::Gray());
// 
//     RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
//     render_scene_config.pushBackLight(std::move(flash_light));
}

void GLApp::drawBackground()
{
	int xmin = -10;
	int xmax = 10;
	int zmin = -10;
	int zmax = 10;

	float s = 1.0f;
	int nSub = 10;
	float sub_s = s / nSub;

	glPushMatrix();

	float ep = 0.0001f;
	glPushMatrix();
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

	//Draw background grid
	glLineWidth(2.0f);
	glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
	glBegin(GL_LINES);
	for (int i = xmin; i <= xmax; i++)
	{
		glVertex3f(i*s, 0, zmin*s);
		glVertex3f(i*s, 0, zmax*s);
	}
	for (int i = zmin; i <= zmax; i++)
	{
		glVertex3f(xmin*s, 0, i*s);
		glVertex3f(xmax*s, 0, i*s);
	}
	
	glEnd();

	glLineWidth(1.0f);
	glLineStipple(1, 0x5555);
	glEnable(GL_LINE_STIPPLE);
	glColor4f(0.55f, 0.55f, 0.55f, 1.0f);
	glBegin(GL_LINES);
	for (int i = xmin; i <= xmax; i++)
	{
		for (int j = 1; j < nSub; j++)
		{
			glVertex3f(i*s + j*sub_s, 0, zmin*s);
			glVertex3f(i*s + j*sub_s, 0, zmax*s);
		}
	}
	for (int i = zmin; i <= zmax; i++)
	{
		for (int j = 1; j < nSub; j++)
		{
			glVertex3f(xmin*s, 0, i*s + j*sub_s);
			glVertex3f(xmax*s, 0, i*s + j*sub_s);
		}
	}
	glEnd();
	glDisable(GL_LINE_STIPPLE);

	glPopMatrix();

	drawAxis();
}

void GLApp::drawAxis()
{
	GLfloat mv[16];
	GLfloat proj[16];
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glGetFloatv(GL_MODELVIEW_MATRIX, mv);
	mv[12] = mv[13] = mv[14] = 0.0;

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glMatrixMode(GL_PROJECTION);
	
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, 0.0, 1.0, 1.0, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadMatrixf(mv);

	//Draw axes
	glViewport(20, 10, 90, 80);
	glColor3ub(255, 255, 255);
	glLineWidth(1.0f);
	const float len = 0.9f;
	GLfloat origin[3] = {0.0f, 0.0f, 0.0f};
	glBegin(GL_LINES);
	glColor3f(1, 0, 0);
	glVertex3f(origin[0], origin[1], origin[2]);
	glVertex3f(origin[0] + len, origin[1], origin[2]);
	glColor3f(0, 1, 0);
	glVertex3f(origin[0], origin[1], origin[2]);
	glVertex3f(origin[0], origin[1] + len, origin[2]);
	glColor3f(0, 0, 1);
	glVertex3f(origin[0], origin[1], origin[2]);
	glVertex3f(origin[0], origin[1], origin[2] + len);
	glEnd();

	// Draw labels
	glColor3f(1, 0, 0);
	glRasterPos3f(origin[0] + len, origin[1], origin[2]);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'x');
	glColor3f(0, 1, 0);
	glRasterPos3f(origin[0], origin[1] + len, origin[2]);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'y');
	glColor3f(0, 0, 1);
	glRasterPos3f(origin[0], origin[1], origin[2] + len);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'z');

	glPopAttrib();

	// Restore viewport, projection and model-view matrices
	glViewport(0, 0, getWidth(), getHeight());
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void GLApp::drawBoundingBox(Vector3f lo, Vector3f hi)
{
	glPushMatrix();

	glColor3f(0.8, 0.8, 0.8);
	glLineWidth(3);
	glBegin(GL_LINES);
	glVertex3f(lo[0], lo[1], lo[2]);
	glVertex3f(hi[0], lo[1], lo[2]);
	glVertex3f(lo[0], lo[1], lo[2]);
	glVertex3f(lo[0], hi[1], lo[2]);
	glVertex3f(lo[0], lo[1], lo[2]);
	glVertex3f(lo[0], lo[1], hi[2]);

	glVertex3f(hi[0], lo[1], lo[2]);
	glVertex3f(hi[0], hi[1], lo[2]);

	glVertex3f(hi[0], lo[1], lo[2]);
	glVertex3f(hi[0], lo[1], hi[2]);

	glVertex3f(lo[0], lo[1], hi[2]);
	glVertex3f(hi[0], lo[1], hi[2]);

	glVertex3f(lo[0], lo[1], hi[2]);
	glVertex3f(lo[0], hi[1], hi[2]);

	glVertex3f(lo[0], hi[1], hi[2]);
	glVertex3f(lo[0], hi[1], lo[2]);

	glVertex3f(lo[0], hi[1], lo[2]);
	glVertex3f(hi[0], hi[1], lo[2]);

	glVertex3f(lo[0], hi[1], hi[2]);
	glVertex3f(hi[0], hi[1], hi[2]);
	glVertex3f(hi[0], lo[1], hi[2]);
	glVertex3f(hi[0], hi[1], hi[2]);
	glVertex3f(hi[0], hi[1], lo[2]);
	glVertex3f(hi[0], hi[1], hi[2]);
	glEnd();

	glPopMatrix();
}

void GLApp::drawString(std::string s, const Color &color, int x, int y)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	gluOrtho2D(0, getWidth(), 0, getHeight());
	glColor3f(color.r, color.g, color.b);
	glRasterPos2i(x, y);

	for (int i = 0; i < (int)s.length(); i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, s[i]);
	}

	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glPopAttrib();

	glViewport(0, 0, getWidth(), getHeight());
}

} //end of namespace dyno
