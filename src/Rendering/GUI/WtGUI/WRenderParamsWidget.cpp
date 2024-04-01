#include "WRenderParamsWidget.h"
#include "WSimulationCanvas.h"

#include <Wt/WPushButton.h>
#include <Wt/WPanel.h>
#include <Wt/WColorPicker.h>
#include <Wt/WDoubleSpinBox.h>
#include <Wt/WLabel.h>
#include <Wt/WTable.h>
#include <Wt/WCheckBox.h>
#include <Wt/WHBoxLayout.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WGridLayout.h>
#include <Wt/WBorderLayout.h>
#include <Wt/WSlider.h>
#include <Wt/WText.h>

#include <GLRenderEngine.h>

using namespace dyno;

WRenderParamsWidget::WRenderParamsWidget(RenderParams* rparams)
{
	this->setLayoutSizeAware(true);
	this->setOverflow(Wt::Overflow::Auto);
	this->setHeight(Wt::WLength("100%"));

	createLightPanel();
	//createCameraPanel();
	createRenderPanel();

	mRenderParams = rparams;
	update();

	// connect signal
	mAmbientColor->colorInput().connect(this, &WRenderParamsWidget::updateRenderParams);
	mAmbientScale->valueChanged().connect(this, &WRenderParamsWidget::updateRenderParams);
	mLightColor->colorInput().connect(this, &WRenderParamsWidget::updateRenderParams);
	mLightScale->valueChanged().connect(this, &WRenderParamsWidget::updateRenderParams);
	mLightTheta->valueChanged().connect(this, &WRenderParamsWidget::updateRenderParams);
	mLightPhi->valueChanged().connect(this, &WRenderParamsWidget::updateRenderParams);

	//mCameraEyeX->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraEyeY->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraEyeZ->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraTargetX->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraTargetY->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraTargetZ->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraUpX->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraUpY->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraUpZ->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraFov->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraAspect->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraClipNear->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mCameraClipFar->changed().connect(this, &WRenderParamsWidget::updateRenderParams);

	mSceneBounds->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	//mAxisHelper->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	mGroundPlane->changed().connect(this, &WRenderParamsWidget::updateRenderParams);
	mGroundScale->valueChanged().connect(this, &WRenderParamsWidget::updateRenderParams);
	mBackgroudColor0->colorInput().connect(this, &WRenderParamsWidget::updateRenderParams);
	mBackgroudColor1->colorInput().connect(this, &WRenderParamsWidget::updateRenderParams);
}

template<class T>
T* addTableRow(Wt::WTable* table, std::string label, int labelWidth = 120, int widgetWidth = 120)
{
	int row = table->rowCount();
	auto cell0 = table->elementAt(row, 0);
	auto cell1 = table->elementAt(row, 1);

	cell0->addNew<Wt::WText>(label);
	cell0->setContentAlignment(Wt::AlignmentFlag::Middle);
	cell0->setWidth(labelWidth);

	cell1->setContentAlignment(Wt::AlignmentFlag::Middle);
	cell1->setWidth(widgetWidth);

	T* widget = cell1->addNew<T>();
	widget->setWidth(widgetWidth);
	return widget;
}

void WRenderParamsWidget::createLightPanel()
{
	// ambient light
	{
		auto panel = this->addNew<Wt::WPanel>();
		panel->setCollapsible(true);
		panel->setTitle("Ambient Light");
		auto table = panel->setCentralWidget(std::make_unique<Wt::WTable>());
		table->setMargin(10);

		// ambient light
		mAmbientColor = addTableRow<Wt::WColorPicker>(table, "Ambient Color");
		mAmbientScale = addTableRow<Wt::WDoubleSpinBox>(table, "Ambient Scale");
		mAmbientScale->setRange(1, 100);		
	}	

	// main directional light
	{
		auto panel = this->addNew<Wt::WPanel>();
		panel->setCollapsible(true);
		panel->setTitle("Main Directional Light");
		auto table = panel->setCentralWidget(std::make_unique<Wt::WTable>());
		table->setMargin(10);

		mLightColor = addTableRow<Wt::WColorPicker>(table, "Light Color");
		mLightScale = addTableRow<Wt::WDoubleSpinBox>(table, "Light Scale");
		mLightScale->setRange(1, 100);

		mLightTheta = addTableRow<Wt::WSlider>(table, "Light Theta");
		mLightPhi = addTableRow<Wt::WSlider>(table, "Light Phi");
		mLightTheta->setRange(0, 180);
		mLightPhi->setRange(-180, 180);
	}

	mAmbientColor->setStyleClass("color-picker");
	mLightColor->setStyleClass("color-picker");

}

void WRenderParamsWidget::createCameraPanel()
{
	// light
	auto panel = this->addNew<Wt::WPanel>();
	panel->setCollapsible(true);
	panel->setTitle("Camera Settings");
	auto table = panel->setCentralWidget(std::make_unique<Wt::WTable>());
	table->setMargin(10);

	mCameraEyeX = addTableRow<Wt::WDoubleSpinBox>(table, "Eye X");
	mCameraEyeY = addTableRow<Wt::WDoubleSpinBox>(table, "Eye Y");
	mCameraEyeZ = addTableRow<Wt::WDoubleSpinBox>(table, "Eye Z");

	mCameraTargetX = addTableRow<Wt::WDoubleSpinBox>(table, "Target X");
	mCameraTargetY = addTableRow<Wt::WDoubleSpinBox>(table, "Target Y");
	mCameraTargetZ = addTableRow<Wt::WDoubleSpinBox>(table, "Target Z");

	mCameraUpX = addTableRow<Wt::WDoubleSpinBox>(table, "Up X");
	mCameraUpY = addTableRow<Wt::WDoubleSpinBox>(table, "Up Y");
	mCameraUpZ = addTableRow<Wt::WDoubleSpinBox>(table, "Up Z");

	mCameraFov = addTableRow<Wt::WDoubleSpinBox>(table, "FOV(Vertical)");
	mCameraAspect = addTableRow<Wt::WDoubleSpinBox>(table, "Aspect");
	mCameraClipNear = addTableRow<Wt::WDoubleSpinBox>(table, "Near Clip");
	mCameraClipFar = addTableRow<Wt::WDoubleSpinBox>(table, "Far Clip");

	table->elementAt(13, 0)->setColumnSpan(2);
	table->elementAt(13, 0)->setContentAlignment(Wt::AlignmentFlag::Center);
	auto updateBtn = table->elementAt(13, 0)->addNew<Wt::WPushButton>("Update from Canvas");
	updateBtn->setMargin(10, Wt::Side::Top);
	updateBtn->clicked().connect(this, &WRenderParamsWidget::update);

	// aspect is auto computed from framebuffer sizer
	mCameraAspect->setEnabled(false);

}


void WRenderParamsWidget::createRenderPanel()
{
	auto panel = this->addNew<Wt::WPanel>();
	panel->setCollapsible(true);
	panel->setTitle("Render Settings");
	auto table = panel->setCentralWidget(std::make_unique<Wt::WTable>());
	table->setMargin(10);

	mSceneBounds = addTableRow<Wt::WCheckBox>(table, "Scene Bound");
	//mAxisHelper = addTableRow<Wt::WCheckBox>(table, "Axis Helper");
	mGroundPlane = addTableRow<Wt::WCheckBox>(table, "Ground Plane");
	mGroundScale = addTableRow<Wt::WSlider>(table, "Ground Scale");
	mGroundScale->setRange(1, 10);
	mBackgroudColor0 = addTableRow<Wt::WColorPicker>(table, "Background");
	mBackgroudColor0->setStyleClass("color-picker");
	mBackgroudColor1 = addTableRow<Wt::WColorPicker>(table, "Background");
	mBackgroudColor1->setStyleClass("color-picker");

}

Wt::WColor Glm2WColor(glm::vec3 v)
{
	return Wt::WColor(v.x * 255, v.y * 255, v.z * 255);
}

glm::vec3 WColor2Glm(Wt::WColor clr)
{
	return { clr.red() / 255.f, clr.green() / 255.f, clr.blue() / 255.f };
}

glm::vec3 xyz2sphere(glm::vec3 v)
{
	float xz = glm::length(glm::vec2(v.x, v.z));
	float theta = atan2(xz, v.y);
	float phi = atan2(v.z, v.x);
	return glm::vec3(theta, phi, glm::length(v));
}

glm::vec3 sphere2xyz(glm::vec3 v)
{
	float r = v.z;
	float x = r * sinf(v.x) * cosf(v.y);
	float y = r * cosf(v.x);
	float z = r * sinf(v.x) * sinf(v.y);
	return glm::vec3(x, y, z);
}

void WRenderParamsWidget::update()
{
	// light
	mAmbientColor->setColor(Glm2WColor(mRenderParams->light.ambientColor));
	mAmbientScale->setValue(mRenderParams->light.ambientScale);
	mLightColor->setColor(Glm2WColor(mRenderParams->light.mainLightColor));
	mLightScale->setValue(mRenderParams->light.mainLightScale);
	//
	glm::vec3 dir = glm::normalize(mRenderParams->light.mainLightDirection);
	glm::vec3 polar = xyz2sphere(dir);
	mLightTheta->setValue(glm::degrees(polar.x));
	mLightPhi->setValue(glm::degrees(polar.y));

	//TODO:
	// camera
// 	mCameraEyeX->setValue(mRenderParams->camera.eye.x);
// 	mCameraEyeY->setValue(mRenderParams->camera.eye.y);
// 	mCameraEyeZ->setValue(mRenderParams->camera.eye.z);
// 	mCameraTargetX->setValue(mRenderParams->camera.target.x);
// 	mCameraTargetY->setValue(mRenderParams->camera.target.y);
// 	mCameraTargetZ->setValue(mRenderParams->camera.target.z);
// 	mCameraUpX->setValue(mRenderParams->camera.up.x);
// 	mCameraUpY->setValue(mRenderParams->camera.up.y);
// 	mCameraUpZ->setValue(mRenderParams->camera.up.z);
// 	mCameraFov->setValue(mRenderParams->camera.y_fov);
// 	mCameraAspect->setValue(mRenderParams->camera.aspect);
// 	mCameraClipNear->setValue(mRenderParams->camera.z_min);
// 	mCameraClipFar->setValue(mRenderParams->camera.z_max);

 	// render
//  	mSceneBounds->setChecked(mRenderParams->showSceneBounds);
//  	//mAxisHelper->setChecked(mRenderParams->showAxisHelper);
//  	mGroundPlane->setChecked(mRenderParams->showGround);
//  	mGroundScale->setValue(mRenderParams->planeScale);
//  	mBackgroudColor0->setColor(Glm2WColor(mRenderParams->bgColor0));
// 	mBackgroudColor1->setColor(Glm2WColor(mRenderParams->bgColor1));
}

void WRenderParamsWidget::updateRenderParams()
{
	mRenderParams->light.ambientColor = WColor2Glm(mAmbientColor->color());
	mRenderParams->light.ambientScale = mAmbientScale->value();

	mRenderParams->light.mainLightColor = WColor2Glm(mLightColor->color());
	mRenderParams->light.mainLightScale = mLightScale->value();

	glm::vec2 polar = glm::radians(glm::vec2(mLightTheta->value(), mLightPhi->value()));
	mRenderParams->light.mainLightDirection = sphere2xyz(glm::vec3(polar, 1));

// 	mRenderParams->camera.eye.x = mCameraEyeX->value();
// 	mRenderParams->camera.eye.y = mCameraEyeY->value();
// 	mRenderParams->camera.eye.z = mCameraEyeZ->value();
// 
// 	mRenderParams->camera.target.x = mCameraTargetX->value();
// 	mRenderParams->camera.target.y = mCameraTargetY->value();
// 	mRenderParams->camera.target.z = mCameraTargetZ->value();
// 
// 	mRenderParams->camera.up.x = mCameraUpX->value();
// 	mRenderParams->camera.up.y = mCameraUpY->value();
// 	mRenderParams->camera.up.z = mCameraUpZ->value();
// 
// 	mRenderParams->camera.y_fov = mCameraFov->value();
// 	//mRenderParams->camera.aspect = mCameraAspect->value();
// 	mRenderParams->camera.z_min = mCameraClipNear->value();
// 	mRenderParams->camera.z_max = mCameraClipFar->value();
// 
//  	mRenderParams->showSceneBounds = mSceneBounds->isChecked();
//  	//mRenderParams->showAxisHelper = mAxisHelper->isChecked();
//  	mRenderParams->showGround = mGroundPlane->isChecked();
//  	mRenderParams->planeScale = mGroundScale->value();
//  	mRenderParams->bgColor0 = WColor2Glm(mBackgroudColor0->color());
// 	mRenderParams->bgColor1 = WColor2Glm(mBackgroudColor1->color());

	mSignal.emit();
}