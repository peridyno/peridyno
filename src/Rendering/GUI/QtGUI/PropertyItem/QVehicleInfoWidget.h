/**
 * Program:   Qt-based widget to Config MultiBody bindings
 * Module:    QVehicleInfoWidget.h
 *
 * Copyright 2026 Yuzhong Guo
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
#include <QScrollArea>
#include <QList>

namespace dyno
{
	class QShapeDetail : public QWidget
	{
		Q_OBJECT
	public:
		QShapeDetail(ShapeConfig shapeData,int id);
		//QShapeDetail::QShapeDetail();

		~QShapeDetail() {}

	signals:
		/**
		 * @brief Transmits a signal when data is updated.
		 */
		void shapeChange();
		void removeShapeItem(int id);
	public slots:
		/**
		 * @brief Updated when any element parameter is changed.
		 */
		void updateData();
		void updateElement(int type);
		void removeItemSlot();

	public:

		ShapeConfig GetShapeConfig() { return mShapeData; };
		void SetShapeConfig(ShapeConfig v) { mShapeData = v; };
		void hideAllWidget()
		{
			//Qt Widgets

			mRadiusWidget->hide();
			mCapsuleLengthWidget->hide();
			mHalfLengthWidget->hide();
			mTetWidget_0->hide();
			mTetWidget_1->hide();
			mTetWidget_2->hide();
			mTetWidget_3->hide();

		}
	private:
		int id = -1;

		ShapeConfig mShapeData;
		QPushButton* mDeleteButton = nullptr;

		QComboBox* mTypeCombox = nullptr;
		//Qt Widgets
		mPiecewiseDoubleSpinBox* mDensity = nullptr;

		mVec3fWidget* mCenterWidget = nullptr;
		mVec3fWidget* mAngleWidget = nullptr;

		mPiecewiseDoubleSpinBox* mRadiusWidget = nullptr;
		mPiecewiseDoubleSpinBox* mCapsuleLengthWidget = nullptr;
		mVec3fWidget* mHalfLengthWidget = nullptr;
		mVec3fWidget* mTetWidget_0 = nullptr;
		mVec3fWidget* mTetWidget_1 = nullptr;
		mVec3fWidget* mTetWidget_2 = nullptr;
		mVec3fWidget* mTetWidget_3 = nullptr;
		QHBoxLayout* mMainLayout = nullptr;

		QHBoxLayout* m_layout;
		const std::vector<ConfigShapeType> mAllShapeType = {
			CONFIG_BOX,
			CONFIG_TET,
			CONFIG_CAPSULE,
			CONFIG_SPHERE,
			CONFIG_TRI,
			CONFIG_COMPOUND,
			CONFIG_Other
		};
	};

	class ShapeDetailListWidget : public QWidget
	{
		Q_OBJECT
	public:
		explicit ShapeDetailListWidget(std::vector<ShapeConfig>* shapes,QWidget* parent = nullptr);

		void setContainerBgColor(const QColor& color);  
		void setContainerBorderColor(const QColor& color);
		void setContainerBorderWidth(int width);         


	signals:
		/**
		 * @brief Transmits a signal when data is updated.
		 */
		void shapesChange();

	public slots:

		void buildShapeDetail(ShapeConfig shapeData, int id);
		void addShapeDetail(bool t);
		void removeLastShapeDetail();
		bool removeShapeDetail(int index);
		void clearAll();
		void updateShapeListData();

	private:

		void initUI();

		QScrollArea* m_scrollArea;       
		QWidget* m_contentWidget;         
		QVBoxLayout* m_contentLayout;     
		QVBoxLayout* m_mainLayout;        
		QPushButton* m_addBtn;            


		QList<QShapeDetail*> m_detailList;
		QColor m_bgColor;                 
		QColor m_borderColor;             

		int m_borderWidth;                

		void updateStyleSheet();

		std::vector<ShapeConfig>* mShapes;
	};



	class QRigidBodyDetail : public QWidget
	{
		Q_OBJECT
	public:
		QRigidBodyDetail(RigidBodyConfig* rigidInfo);
		~QRigidBodyDetail() {}

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
		//Source data
		RigidBodyConfig* mRigidBodyData = nullptr;
		//Widget
		QLineEdit* mNameInput = nullptr;
		QSpinBox* mRigidGroup = nullptr;

		QSpinBox* mVisualMeshID = nullptr;
		mVec3fWidget* mPositionWidget = nullptr;
		mVec3fWidget* mAngleWidget = nullptr;
		mVec3fWidget* mOffsetWidget = nullptr;
		QComboBox* mMotionWidget = nullptr;

		mVec3fWidget* mLinearVelocity = nullptr;
		mVec3fWidget* mAngularVelocity = nullptr;
		

		mVec3fWidget* mInertia1 = nullptr;
		mVec3fWidget* mInertia2 = nullptr;
		mVec3fWidget* mInertia3 = nullptr;

		mPiecewiseDoubleSpinBox* mFriction = nullptr;
		mPiecewiseDoubleSpinBox* mRestitution = nullptr;

		ShapeDetailListWidget* mShapeConfigs;

		QComboBox* mMask;

		std::vector<ConfigMotionType> mAllConfigMotionTypes = {
			CONFIG_Static,
			CONFIG_Kinematic,
			CONFIG_Dynamic,
			CONFIG_NonRotatable,
			CONFIG_NonGravitative
		};

		std::vector<ConfigCollisionMask> mAllConfigCollisionMasks =
		{
			CONFIG_AllObjects,
			CONFIG_BoxExcluded,
			CONFIG_TetExcluded,
			CONFIG_CapsuleExcluded,
			CONFIG_SphereExcluded,
			CONFIG_BoxOnly,
			CONFIG_TetOnly,
			CONFIG_CapsuleOnly,
			CONFIG_SphereOnly,
			CONFIG_Disabled
		};

	};



	class RigidBodyItemLayout : public QVBoxLayout
	{
		Q_OBJECT
	public:
		RigidBodyItemLayout(int id, const RigidBodyConfig& rigidInfo);
		~RigidBodyItemLayout();

		/**
		 * @brief Get current RigidBodyInfo.
		 */
		RigidBodyConfig value();
		/**
		 * @brief Initialization RigidBodyInfo.
		 */
		void setValue(const RigidBodyConfig& v);

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

		ConfigShapeType getType() {return mAllShapeType[mTypeCombox->currentIndex()];}

	signals:
		/**
		 * @brief RigidBody Index in the current list.
		 */
		void removeByElementIndexId(int);

		/**
		 * @brief RigidBody Data Change.
		 */
		void rigidChange(int);

		/**
		 * @brief RigidBody Name Change.
		 */
		void nameChange(int);

	public slots:

		void emitRemove(){emit removeByElementIndexId(mElementIndex);}

		void emitChange(int v){emit rigidChange(v);}

		void emitNameChange(int v) {emit nameChange(v);}

		/**
		 * @brief Create RigidBody Detail Panel.
		 */
		void createRigidDetailWidget();

	public:

		QLineEdit* mNameInput = nullptr;
		RigidBodyConfig mRigidInfo;

	private:

		// Index in the current RigidBody list.
		int mElementIndex = -1;

		// Widgets
		QSpinBox* mShapeIDSpin = nullptr;
		QPushButton* mAddShapeButton = nullptr;
		QPushButton* mRemoveButton = nullptr;
		QPushButton* mEditButton = nullptr;
		QLabel* mIndexLabel = nullptr;
		QComboBox* mTypeCombox = nullptr;
		ShapeDetailListWidget* mShapeList = nullptr;

		//All RigidBody types
		const std::vector<ConfigShapeType> mAllShapeType = { 
			CONFIG_BOX,
			CONFIG_TET,
			CONFIG_CAPSULE,
			CONFIG_SPHERE,
			CONFIG_TRI,
			CONFIG_COMPOUND,
			CONFIG_Other
		};

		//Unique objId used to identify this Item.
		int mObjID = -1;
		std::vector<QRigidBodyDetail*> mDetailWidgets;


	};



	class QJointBodyDetail : public QWidget
	{
		Q_OBJECT
	public:
		QJointBodyDetail(MultiBodyJointConfig& jointInfo);
		~QJointBodyDetail() {}

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

		QCheckBox* mUseMoter = nullptr;
		mPiecewiseDoubleSpinBox* mMoterInput = nullptr;

		MultiBodyJointConfig* mJointData = nullptr;
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
		MultiBodyJointConfig value();
		/**
		 * @brief Initialization RigidBodyInfo.
		 */
		void setValue(const MultiBodyJointConfig& v);
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

		QPushButton* mEditButton = nullptr;
		QPushButton* mRemoveButton = nullptr;

		// All Joint types
		const std::vector<ConfigJointType> mAllJointType = { 
			CONFIG_BallAndSocket,
			CONFIG_Slider, 
			CONFIG_Hinge, 
			CONFIG_Fixed, 
			CONFIG_Point, 
			CONFIG_DistanceJoint,
			CONFIG_OtherJoint 
		};

		MultiBodyJointConfig mJointInfo;
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

		void addRigidBodyItemWidgetByID();

		void addJointItemWidget();

		void removeRigidBodyItemWidgetById(int id);

		void removeJointItemWidgetById(int id);

	private:

		void setJointNameComboxEmpty(mJointItemLayout* item, bool emitSignal, int select, int index);

		int getRigidItemObjID(std::string str);

		void updateJointLayoutComboxText(mJointItemLayout* itemLayout);

		void buildItemWidget(const RigidBodyConfig& rigidBody);

		void createJointItemWidget(const MultiBodyJointConfig& rigidBody);

		void connectJointWidgetSignal(mJointItemLayout* itemLayout);

		void connectRigidWidgetSignal(RigidBodyItemLayout* itemLayout);

	private:
		//Field
		MultiBodyBind mVec;

		//Qt Widgets
		QVBoxLayout* mMainLayout = nullptr;
		QVBoxLayout* mRigidsLayout = nullptr;
		QVBoxLayout* mJointsLayout = nullptr;
		QList<RigidBodyItemLayout*> mRigidBodyItems;
		std::vector<mJointItemLayout*> mJointItems;

		//QueryMap for automatic renaming
		std::map<std::string, int> mName2RigidId;
		std::map<int, std::string> mObjID2Name;

		int mRigidCounter = 0;	//RigidItems Counter 
	};



}
