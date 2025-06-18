#pragma once

#include <Wt/WColorPicker.h>
#include <Wt/WContainerWidget.h>
#include <Wt/WSignal.h>

namespace dyno
{
	struct RenderParams;
};

class WRenderParamsWidget : public Wt::WContainerWidget
{
public:
	WRenderParamsWidget(dyno::RenderParams* rparams);

	void update();
	Wt::Signal<>& valueChanged() { return mSignal; };

private:
	void createLightPanel();
	void createCameraPanel();
	void createRenderPanel();

	void updateRenderParams();

private:
	Wt::Signal<>	mSignal;
	dyno::RenderParams*	mRenderParams;

	// ambient illumination
	Wt::WColorPicker*	mAmbientColor;
	Wt::WDoubleSpinBox*	mAmbientScale;

	// main directional color
	Wt::WColorPicker*   mLightColor;
	Wt::WDoubleSpinBox*	mLightScale;
	Wt::WSlider*		mLightTheta;
	Wt::WSlider*		mLightPhi;

	// camera
	Wt::WDoubleSpinBox* mCameraEyeX;
	Wt::WDoubleSpinBox* mCameraEyeY;
	Wt::WDoubleSpinBox* mCameraEyeZ;
	
	Wt::WDoubleSpinBox* mCameraTargetX;
	Wt::WDoubleSpinBox* mCameraTargetY;
	Wt::WDoubleSpinBox* mCameraTargetZ;

	Wt::WDoubleSpinBox* mCameraUpX;
	Wt::WDoubleSpinBox* mCameraUpY;
	Wt::WDoubleSpinBox* mCameraUpZ;

	Wt::WDoubleSpinBox* mCameraFov;
	Wt::WDoubleSpinBox* mCameraAspect;
	Wt::WDoubleSpinBox* mCameraClipNear;
	Wt::WDoubleSpinBox* mCameraClipFar;

	// render
	Wt::WCheckBox*		mSceneBounds;
	//Wt::WCheckBox*		mAxisHelper;
	Wt::WCheckBox*		mGroundPlane;
	Wt::WSlider*		mGroundScale;
	Wt::WColorPicker*	mBackgroudColor0;
	Wt::WColorPicker*	mBackgroudColor1;
};