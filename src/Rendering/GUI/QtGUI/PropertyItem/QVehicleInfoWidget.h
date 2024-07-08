/**
 * Program:   Qt-based widget to visualize Vec3f or Vec3d
 * Module:    QVector3FieldWidget.h
 *
 * Copyright 2023 Xiaowei He
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
#include "VehicleInfo.h"
#include <QComboBox>
#include <QLineEdit>
#include "qcheckbox.h"


namespace dyno
{

	class mRigidBodyItemLayout : public QHBoxLayout
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET
		mRigidBodyItemLayout(int id)
		{
			this->setContentsMargins(0, 0, 0, 0);

			mId = id;
			index = new QLabel(std::to_string(id).c_str());		
			nameInput = new QLineEdit;
			typeCombox = new QComboBox;
				
			for (ConfigShapeType it : VecShapeType)
			{
				switch (it)
				{
				case dyno::ConfigShapeType::Box:
					typeCombox->addItem("Box");
					break;
				case dyno::ConfigShapeType::Tet:
					typeCombox->addItem("Tet");
					break;
				case dyno::ConfigShapeType::Capsule:
					typeCombox->addItem("Capsule");
					break;
				case dyno::ConfigShapeType::Sphere:
					typeCombox->addItem("Sphere");
					break;
				case dyno::ConfigShapeType::Tri:
					typeCombox->addItem("Tri");
					break;
				case dyno::ConfigShapeType::OtherShape:
					typeCombox->addItem("Other");
					break;
				default:
					break;
				}
			}

			ShapeIDSpin = new QSpinBox;
			ShapeIDSpin->setRange(-1,200);
			ShapeIDSpin->setValue(-1);

			//RigidBodyIDSpin = new QSpinBox;
			//RigidBodyIDSpin->setValue(0);

			removeButton = new QPushButton("Delete");
			offsetButton = new QPushButton("Edit Offset");

			this->addWidget(index, 0);
			this->addWidget(nameInput, 0);
			//this->addWidget(RigidBodyIDSpin, 0);
			this->addWidget(ShapeIDSpin, 0);
			this->addWidget(typeCombox, 0);
			this->addWidget(offsetButton, 0);
			this->addWidget(removeButton, 0);

			index->setFixedWidth(25);
			nameInput->setFixedWidth(100);
			typeCombox->setFixedWidth(76);
			//RigidBodyIDSpin->setFixedWidth(76);
			ShapeIDSpin->setFixedWidth(76);
			offsetButton->setFixedWidth(76);
			removeButton->setFixedWidth(76);

			//this->layout()->addWidget(index);
			QObject::connect(nameInput, &QLineEdit::editingFinished, [=]() {emitChange(1);});
			QObject::connect(nameInput, &QLineEdit::editingFinished, [=]() {emitNameChange(1); });


			QObject::connect(ShapeIDSpin, SIGNAL(valueChanged(int)), this, SLOT(emitChange(int)));
			//QObject::connect(RigidBodyIDSpin, SIGNAL(valueChanged(int)), this, SLOT(emitChange(int)));
			QObject::connect(typeCombox, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));
			//offset 偏移更新未完成

			QObject::connect(removeButton, SIGNAL(pressed()), this, SLOT(emitRemoveSignal()));

			//this->RigidBodyIDSpin->setValue(mId);

			this->nameInput->setText(std::string("Rigid").append(std::to_string(mId)).c_str());
			this->typeCombox->setCurrentIndex(2);
			//emitNameChange(1);
		};


		~mRigidBodyItemLayout()
		{
			delete ShapeIDSpin;
			delete removeButton;
			delete index;
			delete offsetButton;
			delete nameInput;
			delete typeCombox;
			//delete RigidBodyIDSpin;
		};

		VehicleRigidBodyInfo value() 
		{
			rigidInfo.shapeName.name = nameInput->text().toStdString();
			rigidInfo.shapeName.rigidId = mId;

	
			rigidInfo.meshShapeId = ShapeIDSpin->value();
			rigidInfo.shapeType = VecShapeType[typeCombox->currentIndex()];
			rigidInfo.offsetTransform = offset;

			return rigidInfo;
		};


		void setValue(const VehicleRigidBodyInfo& v) 
		{ 
			nameInput->setText(QString(v.shapeName.name.c_str()));
			//RigidBodyIDSpin->setValue(v.shapeName.shapeId);
			ShapeIDSpin->setValue(v.meshShapeId);
			for (size_t i = 0; i < VecShapeType.size(); i++)
			{
				if (VecShapeType[i] == v.shapeType)
					typeCombox->setCurrentIndex(i);
			}
			offset = v.offsetTransform;
		}

		void setId(int id) { mId = id; index->setText(std::to_string(id).c_str()); };

		void setObjId(int id) { objID = id; };

		int getObjID() { return objID; };
		int getRigidID() { return mId; };

	signals:
		void removeById(int);
		void valueChange(int);
		void nameChange(int);

	public slots:
		void emitRemoveSignal()
		{
			emit removeById(mId);
		}
		void emitChange(int v)
		{
			emit valueChange(v);
		}
		void emitNameChange(int v) 
		{
			emit nameChange(v);
		}
		
	public:

		QLineEdit* nameInput = nullptr;

	private:

		int mId = -1;

		//QSpinBox* RigidBodyIDSpin = nullptr;
		QSpinBox* ShapeIDSpin = nullptr;
		QPushButton* removeButton = nullptr;
		QPushButton* offsetButton = nullptr;
		QLabel* index = nullptr;

		QComboBox* typeCombox = nullptr;
		Transform3f offset = Transform3f();

		const std::vector<ConfigShapeType> VecShapeType = { Box,Tet,Capsule,Sphere ,Tri, OtherShape };

		VehicleRigidBodyInfo rigidInfo; 

		int objID = -1;
	};


	class QRigidBodyModify : public QWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET
		QRigidBodyModify(mRigidBodyItemLayout* parent)
		{
			this->setContentsMargins(0, 0, 0, 0);


		}

	private:

		QSpinBox* ShapeIDSpin = nullptr;

	};


	class mJointItemLayout : public QHBoxLayout
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET
		mJointItemLayout(int id)
		{
			this->setContentsMargins(0, 0, 0, 0);

			mId = id;
			index = new QLabel(std::to_string(id).c_str());

			nameInput1 = new QComboBox;
			nameInput2 = new QComboBox;
			typeInput = new QComboBox;
				
			useMoter = new QCheckBox;
			MoterInput = new QDoubleSpinBox;
			anchorOffsetButton = new QPushButton("Offset");
			rangeButton = new QPushButton("Range");
			axisButton = new QPushButton("Axis");


			for (ConfigJointType it : vecJointType)
			{
				//					typeInput->addItem("Box");
				switch (it)
				{
				case dyno::BallAndSocket:
					typeInput->addItem("Ball");
					break;
				case dyno::Slider:
					typeInput->addItem("Slider");
					break;
				case dyno::Hinge:
					typeInput->addItem("Hinge");
					break;
				case dyno::Fixed:
					typeInput->addItem("Fixed");
					break;
				case dyno::Point:
					typeInput->addItem("Point");
					break;
				case dyno::OtherJoint:
					typeInput->addItem("Other");
					break;
				default:
					break;
				}
			}

			removeButton = new QPushButton("Delete");

			this->addWidget(index, 0);
			this->addWidget(nameInput1, 0);
			this->addWidget(nameInput2, 0);
			this->addWidget(typeInput, 0);
			this->addWidget(useMoter, 0);
			this->addWidget(MoterInput, 0);
			this->addWidget(anchorOffsetButton, 0);
			this->addWidget(rangeButton, 0);
			this->addWidget(axisButton, 0);
			this->addWidget(removeButton, 0);

			index->setFixedWidth(25);
			nameInput1->setFixedWidth(90);
			nameInput2->setFixedWidth(90);
			typeInput->setFixedWidth(65);
			useMoter->setFixedWidth(20);
			MoterInput->setFixedWidth(45);
			anchorOffsetButton->setFixedWidth(50);
			rangeButton->setFixedWidth(45);
			axisButton->setFixedWidth(45);
			axisButton->setFixedWidth(45);
			removeButton->setFixedWidth(75);

			//this->layout()->addWidget(index);

			QObject::connect(nameInput1, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));
			QObject::connect(nameInput2, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));


			QObject::connect(nameInput2, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));

			QObject::connect(useMoter, SIGNAL(stateChanged(int)), this, SLOT(emitChange(int)));

			QObject::connect(MoterInput, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double value)
				{
					int intValue = static_cast<int>(value); 

					emitChange(intValue);
				});

			QObject::connect(typeInput, SIGNAL(currentIndexChanged(int)), this, SLOT(emitChange(int)));
			//offset 偏移更新未完成
			//CheckBox 未完成更新

			QObject::connect(removeButton, SIGNAL(pressed()), this, SLOT(emitRemoveSignal()));

			this->typeInput->setCurrentIndex(2);
		};


		~mJointItemLayout()
		{
			delete index;

			delete nameInput1;
			delete nameInput2;

			delete anchorOffsetButton;
			delete rangeButton;
			delete axisButton;

			delete useMoter;
			delete MoterInput;
			delete typeInput;
			delete removeButton;
		};

		VehicleJointInfo value()
		{
			//jointInfo.Joint_Actor = ActorId;
			jointInfo.JointName1.name = nameInput1->currentText().toStdString();
			jointInfo.JointName2.name = nameInput2->currentText().toStdString();

			jointInfo.type = vecJointType[typeInput->currentIndex()];
			jointInfo.useMoter = useMoter->isChecked();
			jointInfo.useRange = useRange;

			jointInfo.d_min = d_min;
			jointInfo.d_max = d_max;
			jointInfo.v_moter = MoterInput->value();
			jointInfo.Axis = Axis;

			return jointInfo;
		};


		void setValue(const VehicleJointInfo& v)
		{
			name1_ObjID = (v.JointName1.rigidId);
			name2_ObjID = (v.JointName2.rigidId);

			//Type
			useMoter->setChecked(v.useMoter);
			useRange = jointInfo.useRange;
			anchorPoint = v.anchorPoint;
			d_min = v.d_min;
			d_max = v.d_max;
			MoterInput->setValue(v.v_moter);
			Axis = v.Axis;
		}



		void setId(int id) { mId = id; index->setText(std::to_string(id).c_str()); };

	signals:
		void removeById(int);
		void valueChange(int);


	public slots:
		void emitRemoveSignal()
		{
			emit removeById(mId);
		}
		void emitChange(int v)
		{
			emit valueChange(v);
		}

	public:
		


	public:

		QComboBox* nameInput1 = nullptr;
		QComboBox* nameInput2 = nullptr;

		int name1_ObjID = -1;
		int name2_ObjID = -1;



	private:

		int mId = -1;
		//Vec2i ActorId = Vec2i(-1, -1);
		QComboBox* typeInput = nullptr;
		QLabel* index = nullptr;

		QCheckBox* useMoter = nullptr;
		QDoubleSpinBox* MoterInput = nullptr;

		QPushButton* anchorOffsetButton = nullptr;
		QPushButton* rangeButton = nullptr;
		QPushButton* axisButton = nullptr;
		QPushButton* removeButton = nullptr;

		const std::vector<ConfigJointType> vecJointType = { BallAndSocket,Slider,Hinge,Fixed,Point,OtherJoint };

		// update By Button
		bool useRange = false;
		Vector<Real, 3> anchorPoint;

		Real d_min;
		Real d_max;
		Vector<Real, 3> Axis;


		VehicleJointInfo jointInfo;
	};


	class QVehicleInfoWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QVehicleInfoWidget(FBase* field);

		~QVehicleInfoWidget() override {};

	signals:
		void vectorChange();


	public slots:
		//Called when the widget is updated

		void updateField()
		{
			FVar<VehicleBind>* f = TypeInfo::cast<FVar<VehicleBind>>(field());
			if (f != nullptr)
			{
				f->setValue(mVec);
			}

			printField();
		};


		void updateJointComboBox()
		{
			//从rigidItem更新数据
			buildMap();



			for (auto itemLayout : allJointItem)
			{


				//更新到Joint的界面上
				updateJointLayoutShapeName(itemLayout);

				//set当前INDEX
				itemLayout->nameInput1->blockSignals(true);
				itemLayout->nameInput2->blockSignals(true);

				if (itemLayout->name1_ObjID != -1) 
				{
					auto str = m_objID_Name[itemLayout->name1_ObjID];
					if (str == std::string("")) 
					{
						setJointNameCurrentEmpty(itemLayout, true, 1, -1);
					}
					else 
					{
						auto index = m_Name_RigidId[str];
						itemLayout->nameInput1->setCurrentIndex(index);
					}
				}

				if (itemLayout->name2_ObjID != -1)
				{
					auto str = m_objID_Name[itemLayout->name2_ObjID];
					if (str == std::string(""))
					{
						setJointNameCurrentEmpty(itemLayout, true, 2, -1);
					}
					else 
					{
						auto index = m_Name_RigidId[str];
						itemLayout->nameInput2->setCurrentIndex(index);
					}
				}


				if (itemLayout->name1_ObjID == -1)
				{
					setJointNameCurrentEmpty(itemLayout, true, 1, -1);
					printf("set Index : %d\n", -1);
				}

				if (itemLayout->name2_ObjID == -1)
				{
					setJointNameCurrentEmpty(itemLayout, true, 2, -1);
					printf("set Index : %d\n", -1);
				}
				itemLayout->nameInput1->blockSignals(false);
				itemLayout->nameInput2->blockSignals(false);

			}

		
			updateJoint_RigidObjID();

		}

		void setJointNameCurrentEmpty(mJointItemLayout* item,bool emitSignal,int select,int index)
		{
			if (select == 1)
			{
				if (emitSignal) 
					item->nameInput1->blockSignals(false);
				else
					item->nameInput1->blockSignals(true);
				item->nameInput1->setCurrentIndex(index);
			}
			else if(select == 2)
			{
				if (emitSignal)
					item->nameInput2->blockSignals(false);
				else
					item->nameInput2->blockSignals(true);
				item->nameInput2->setCurrentIndex(index);
			}

			item->nameInput1->blockSignals(false);
			item->nameInput2->blockSignals(false);

			
		};

		//Called when the field is updated
		void updateWidget();



		void updateVector(int) { updateVector(); }

		void updateVector()
		{		


			//*******************************  UpdateData  *******************************//

			mVec.vehicleRigidBodyInfo.clear();
			for (size_t i = 0; i < allItem.size(); i++)
			{
				mVec.vehicleRigidBodyInfo.push_back(allItem[i]->value());
			}

			//*******************************  BuildMap  *******************************//


			buildMap();

			//*******************************  UpdateData  *******************************//
			mVec.vehicleJointInfo.clear();
			
			//update Rigid ID
			for (size_t i = 0; i < allJointItem.size(); i++)
			{
				auto& jointItem = allJointItem[i];
				mVec.vehicleJointInfo.push_back(jointItem->value());
				auto& jointInfo = mVec.vehicleJointInfo[i];
				
				if(jointInfo.JointName1.name!=std::string(""))
					jointInfo.JointName1.rigidId = m_Name_RigidId[jointInfo.JointName1.name];

				if (jointInfo.JointName2.name != std::string(""))
					jointInfo.JointName2.rigidId = m_Name_RigidId[jointInfo.JointName2.name];
			}


			emit vectorChange();
			
		}

		void buildMap() 
		{
			m_objID_Name.clear();
			m_Name_RigidId.clear();

			for (auto it : allItem)
			{
				//用objID查询名称。
				m_objID_Name[it->getObjID()] = it->nameInput->text().toStdString();
				//用名称查询RigidID。
				m_Name_RigidId[it->nameInput->text().toStdString()] = it->getRigidID();
			}

		}

		void updateJoint_RigidObjID() 
		{
			for (size_t i = 0; i < allJointItem.size(); i++)
			{
				auto& jointItem = allJointItem[i];
				jointItem->name1_ObjID = getRigidItemObjID(jointItem->nameInput1->currentText().toStdString());
				std::cout << jointItem->nameInput1->currentText().toStdString() << ", " << jointItem->name1_ObjID << std::endl;
				jointItem->name2_ObjID = getRigidItemObjID(jointItem->nameInput2->currentText().toStdString());

			}
		}

		void addItemWidget()
		{
			addItemWidgetByID(counter_RigidID);
			counter_RigidID++;
		}

		void addItemWidgetByID(int objId) 
		{
			mRigidBodyItemLayout* itemLayout = new mRigidBodyItemLayout(allItem.size());
			itemLayout->setObjId(objId);

			connectRigidWidgetSignal(itemLayout);


			rigidsLayout->addLayout(itemLayout);
			allItem.push_back(itemLayout);

			updateVector();
		}



		void removeItemWidgetById(int id)
		{
			rigidsLayout->removeItem(allItem[id]);
			delete allItem[id];
			allItem.erase(allItem.begin() + id);
			for (size_t i = 0; i < allItem.size(); i++)
			{
				allItem[i]->setId(i);
			}

			//*******************************  UpdateData  *******************************//

			updateVector();
			buildMap();
			updateJointComboBox();		
		}


		void addJointItemWidget()
		{
			mJointItemLayout* itemLayout = new mJointItemLayout(allJointItem.size());

			connectJointWidgetSignal(itemLayout);

			jointsLayout->addLayout(itemLayout);
			allJointItem.push_back(itemLayout);

			updateJointLayoutShapeName(itemLayout);
		
			updateVector();
		}

		void removeJointItemWidgetById(int id)
		{
			jointsLayout->removeItem(allJointItem[id]);
			delete allJointItem[id];
			allJointItem.erase(allJointItem.begin() + id);
			for (size_t i = 0; i < allJointItem.size(); i++)
			{
				allJointItem[i]->setId(i);
			}

			updateVector();
		}



	private:

		int getRigidItemObjID(std::string str)
		{
			for (const auto& pair : m_objID_Name) {
				if (pair.second == str) {
					return  pair.first;
				}
			}
			return -1;
		}

		void updateJointLayoutShapeName(mJointItemLayout* itemLayout)
		{
			itemLayout->nameInput1->blockSignals(true);
			itemLayout->nameInput2->blockSignals(true);

			itemLayout->nameInput1->clear();
			itemLayout->nameInput2->clear();

			for (auto it : m_objID_Name)
			{
				auto name = it.second;
				itemLayout->nameInput1->addItem(name.c_str());
				itemLayout->nameInput2->addItem(name.c_str());
			}

			itemLayout->nameInput1->setCurrentIndex(-1);
			itemLayout->nameInput2->setCurrentIndex(-1);

			itemLayout->nameInput1->blockSignals(false);
			itemLayout->nameInput2->blockSignals(false);
		}

		void createItemWidget(const VehicleRigidBodyInfo& rigidBody)
		{
			{
				mRigidBodyItemLayout* itemLayout = new mRigidBodyItemLayout(allItem.size());
				itemLayout->setObjId(counter_RigidID);
				counter_RigidID++;

				itemLayout->setValue(rigidBody);

				connectRigidWidgetSignal(itemLayout);


				rigidsLayout->addLayout(itemLayout);
				allItem.push_back(itemLayout);
			};
		}

		void createJointItemWidget(const VehicleJointInfo& rigidBody)
		{
			
			{
				mJointItemLayout* itemLayout = new mJointItemLayout(allJointItem.size());


				itemLayout->setValue(rigidBody);

				connectJointWidgetSignal(itemLayout);

				jointsLayout->addLayout(itemLayout);
				allJointItem.push_back(itemLayout);
			};
		}


		void connectJointWidgetSignal(mJointItemLayout* itemLayout)
		{
			QObject::connect(itemLayout, SIGNAL(removeById(int)), this, SLOT(removeJointItemWidgetById(int)));
			QObject::connect(itemLayout, SIGNAL(valueChange(int)), this, SLOT(updateVector()));

			QObject::connect(itemLayout->nameInput1, SIGNAL(currentIndexChanged(int)), this, SLOT(updateJoint_RigidObjID()));
			QObject::connect(itemLayout->nameInput2, SIGNAL(currentIndexChanged(int)), this, SLOT(updateJoint_RigidObjID()));

		}

		void connectRigidWidgetSignal(mRigidBodyItemLayout* itemLayout)
		{
			QObject::connect(itemLayout, SIGNAL(removeById(int)), this, SLOT(removeItemWidgetById(int)));

			QObject::connect(itemLayout, SIGNAL(nameChange(int)), this, SLOT(updateJointComboBox()));
			QObject::connect(itemLayout, SIGNAL(valueChange(int)), this, SLOT(updateVector()));

			QObject::connect(itemLayout->nameInput, SIGNAL(editingFinished()), this, SLOT(buildMap()));
		}

		void printField()
		{
			printf("Current Vehicle Config:\n");
			printf("Rigid Info:\n");
			for (auto it: mVec.vehicleRigidBodyInfo)
			{
				std::cout <<
					it.shapeName.name << ", " << it.shapeName.rigidId << ", " <<
					"Rigid Type : "<<it.shapeType << ", " <<
					"Mesh ShapeID : "<<it.meshShapeId << ", " <<
					"Transform : " << it.offsetTransform.translation()[0] << ", " << it.offsetTransform.translation()[1] << ", " << it.offsetTransform.translation()[2] << ", " <<
					std::endl << "  ***********************************  " << std::endl;
			}

			printf("Joint Info:\n");
			for (auto it : mVec.vehicleJointInfo)
			{
				std::cout <<
					it.JointName1.name << ", " << it.JointName1.rigidId << ", " <<
					it.JointName2.name << ", " << it.JointName2.rigidId << ", " <<
					"Joint Type :" << it.type << ", " <<
					"Moter : " << it.useMoter << ", " << it.v_moter <<
					"Range : " << it.useRange << ", " << it.d_min << " , " << it.d_max << 
					std::endl<<"  ***********************************  " << std::endl;
			}



		}







	private:

		//std::vector<int> mVec;
		VehicleBind mVec;

		QVBoxLayout* mainLayout = nullptr;
		QVBoxLayout* rigidsLayout = nullptr;
		QVBoxLayout* jointsLayout = nullptr;

		std::vector<mRigidBodyItemLayout*> allItem;

		std::vector<mJointItemLayout*> allJointItem;

		std::map<std::string, int> m_Name_RigidId;
		std::map<int, std::string> m_objID_Name;

		int counter_RigidID = 0;	//Counter RigidItem ID
	};



}
