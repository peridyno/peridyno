/**
 * Program:   Qt-based widget to Config Vehicle bindings
 * Module:    QVehicleInfoWidget.h
 *
 * Copyright 2023 Yuzhong Guo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "QFieldWidget.h"
#include "QtGUI/PPropertyWidget.h"
#include "Field.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include "Field/VehicleInfo.h"
#include <QComboBox>
#include <QLineEdit>
#include "qcheckbox.h"
#include "QPiecewiseDoubleSpinBox.h"
#include "qpushbutton.h"
#include "qspinbox.h"


namespace dyno
{


	class QRigidBodyDetail : public QWidget
	{
		Q_OBJECT
	public:
		QRigidBodyDetail(VehicleRigidBodyInfo& rigidInfo);
		~QRigidBodyDetail(){}

	signals:
		/**
		 * @brief Transmits a signal when data is updated.
		 */
		void rigidChange();

	public slots:
		/**
		 * @brief Updated when any element parameter is changed.
		 */
		void updateData();

	private:
		//Current rigidbody type
		ConfigShapeType mCurrentType;

		//Qt Widgets
		mVec3fWidget* mTranslationWidget = nullptr;
		mVec3fWidget* mRotationWidget = nullptr;
		mVec3fWidget* mScaleWidget = nullptr;
		mVec3fWidget* mOffsetWidget = nullptr;
		mPiecewiseDoubleSpinBox* mRadiusWidget = nullptr;
		mPiecewiseDoubleSpinBox* mCapsuleLengthWidget = nullptr;
		mVec3fWidget* mHalfLengthWidget = nullptr;
		mVec3fWidget* mTetWidget_0 = nullptr;
		mVec3fWidget* mTetWidget_1 = nullptr;
		mVec3fWidget* mTetWidget_2 = nullptr;
		mVec3fWidget* mTetWidget_3 = nullptr;
		QComboBox* mMotionWidget = nullptr;	
		QSpinBox* mRigidGroup = nullptr;

		//Source data
		VehicleRigidBodyInfo* mRigidBodyData = nullptr;
		//
		std::vector<ConfigMotionType> mAllConfigMotionTypes = { CMT_Static,CMT_Kinematic,CMT_Dynamic };
	};


	class QJointBodyDetail : public QWidget
	{
		Q_OBJECT
	public:
		QJointBodyDetail(VehicleJointInfo& jointInfo);
		~QJointBodyDetail(){}

	signals:
		/**
		 * @brief Transmits a signal when data is updated.
		 */
		void jointChange();

	public slots:
		/**
		 * @brief Updated when any element parameter is changed.
		 */
		void updateData();


	private:
		ConfigJointType mCurrentType;

		mVec3fWidget* mAnchorPointWidget = nullptr;
		mVec3fWidget* mAxisWidget = nullptr;
		QCheckBox* mUseRangeWidget = nullptr;
		QPiecewiseDoubleSpinBox* mMinWidget = nullptr;
		QPiecewiseDoubleSpinBox* mMaxWidget = nullptr;
		QToggleLabel* mNameLabel = nullptr; 

		VehicleJointInfo* mJointData = nullptr;
	};



	class RigidBodyItemLayout : public QHBoxLayout
	{
		Q_OBJECT
	public:
		RigidBodyItemLayout(int id);
		~RigidBodyItemLayout();

		/**
		 * @brief Get current RigidBodyInfo.
		 */
		VehicleRigidBodyInfo value();
		/**
		 * @brief Initialization RigidBodyInfo.
		 */
		void setValue(const VehicleRigidBodyInfo& v);

		/**
		 * @brief Index in the current RigidBody list.
		 */
		void setId(int id) { mElementIndex = id; mIndexLabel->setText(std::to_string(id).c_str()); };

		/**
		 * @brief Unique objId used to identify this Item.
		 */
		void setObjId(int id) { mObjID = id; };
	
		/**
		 * @brief Unique objId used to identify this Item.
		 */
		int getObjID() { return mObjID; };

		/**
		 * @brief RigidBody Index in the current list.
		 */
		int getRigidID() { return mElementIndex; };

		ConfigShapeType getType() {return mVecShapeType[mTypeCombox->currentIndex()];}

	signals:
		/**
		 * @brief RigidBody Index in the current list.
		 */
		void removeByElementIndexId(int);

		/**
		 * @brief RigidBody Data Change.
		 */
		void valueChange(int);

		/**
		 * @brief RigidBody Name Change.
		 */
		void nameChange(int);

	public slots:

		void emitRemove(){emit removeByElementIndexId(mElementIndex);}

		void emitChange(int v){emit valueChange(v);}

		void emitNameChange(int v) {emit nameChange(v);}

		/**
		 * @brief Create RigidBody Detail Panel.
		 */
		void createRigidDetailWidget();

	public:

		QLineEdit* mNameInput = nullptr;
		VehicleRigidBodyInfo mRigidInfo;

	private:

		// Index in the current RigidBody list.
		int mElementIndex = -1;

		// Widgets
		QSpinBox* mShapeIDSpin = nullptr;
		QPushButton* mRemoveButton = nullptr;
		QPushButton* mOffsetButton = nullptr;
		QLabel* mIndexLabel = nullptr;
		QComboBox* mTypeCombox = nullptr;

		//All RigidBody types
		const std::vector<ConfigShapeType> mVecShapeType = { Box,Tet,Capsule,Sphere ,Tri, OtherShape };
		//Unique objId used to identify this Item.
		int mObjID = -1;
		std::vector<QRigidBodyDetail*> mDetailWidgets;


	};


	class mJointItemLayout : public QHBoxLayout
	{
		Q_OBJECT
	public:
		mJointItemLayout(int id);
	
		~mJointItemLayout();
		/**
		 * @brief Get current RigidBodyInfo.
		 */
		VehicleJointInfo value();
		/**
		 * @brief Initialization RigidBodyInfo.
		 */
		void setValue(const VehicleJointInfo& v);
		/**
		 * @brief Index in the current Joint list.
		 */
		void setId(int id) { mElementIndex = id; mIndex->setText(std::to_string(id).c_str()); };

	signals:
		void removeByElementIndexId(int);
		/**
		 * @brief Joint Data Change.
		 */
		void valueChange(int);


	public slots:

		void emitRemove(){emit removeByElementIndexId(mElementIndex);}

		void emitChange(int v){emit valueChange(v);}

		/**
		 * @brief Create Joint Detail Panel.
		 */
		void createJointDetailWidget();

	public:

		QComboBox* mNameInput1 = nullptr;
		QComboBox* mNameInput2 = nullptr;

		int mName1_ObjID = -1;
		int mName2_ObjID = -1;

	private:
		// Index in the current Joint list.
		int mElementIndex = -1;

		// Qt Widget
		QComboBox* mTypeInput = nullptr;
		QLabel* mIndex = nullptr;
		QCheckBox* mUseMoter = nullptr;
		QDoubleSpinBox* mMoterInput = nullptr;
		QPushButton* mEditButton = nullptr;
		QPushButton* mRemoveButton = nullptr;

		// All Joint types
		const std::vector<ConfigJointType> mVecJointType = { BallAndSocket,Slider,Hinge,Fixed,Point,OtherJoint };
		VehicleJointInfo mJointInfo;
		std::vector<QJointBodyDetail*> mDetailWidgets;
	};


	class QVehicleInfoWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QVehicleInfoWidget(FBase* field);

		~QVehicleInfoWidget();

	signals:
		/**
		 * @brief Data Change.
		 */
		void vectorChange();


	public slots:
		/**
		 * @brief Called when the widget is updated.
		 */
		void updateField();
		/**
		 * @brief Called when the field is updated.
		 */
		void updateWidget();
		/**
		 * @brief Update text and key of the comboxbox
		 */
		void updateJointComboBox();
		/**
		 * @brief Update Configuration Data;
		 */
		void updateVector(int) { updateVector(); }
		void updateVector();

		void bulidQueryMap();

		void updateJoint_RigidObjID();

		void addRigidBodyItemWidget();

		void addRigidBodyItemWidgetByID(int objId);

		void addJointItemWidget();

		void removeRigidBodyItemWidgetById(int id);

		void removeJointItemWidgetById(int id);

	private:

		void setJointNameComboxEmpty(mJointItemLayout* item, bool emitSignal, int select, int index);

		int getRigidItemObjID(std::string str);

		void updateJointLayoutComboxText(mJointItemLayout* itemLayout);

		void createItemWidget(const VehicleRigidBodyInfo& rigidBody);

		void createJointItemWidget(const VehicleJointInfo& rigidBody);

		void connectJointWidgetSignal(mJointItemLayout* itemLayout);

		void connectRigidWidgetSignal(RigidBodyItemLayout* itemLayout);

	private:
		//Field
		VehicleBind mVec;

		//Qt Widgets
		QVBoxLayout* mMainLayout = nullptr;
		QVBoxLayout* mRigidsLayout = nullptr;
		QVBoxLayout* mJointsLayout = nullptr;
		std::vector<RigidBodyItemLayout*> mRigidBodyItems;
		std::vector<mJointItemLayout*> mJointItems;

		//QueryMap for automatic renaming
		std::map<std::string, int> mName2RigidId;
		std::map<int, std::string> mObjID2Name;

		int mRigidCounter = 0;	//RigidItems Counter 
	};



}
