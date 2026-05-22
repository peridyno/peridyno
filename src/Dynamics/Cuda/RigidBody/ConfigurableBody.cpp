#include "ConfigurableBody.h"

#include "Module/CarDriver.h"

//Collision
#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionTriangleSet.h"

//RigidBody
#include "Module/ContactsUnion.h"
#include "Module/TJConstraintSolver.h"
#include "Module/InstanceTransform.h"
#include "Module/SharedFuncsForRigidBody.h"

//Rendering
#include "Module/GLPhotorealisticInstanceRender.h"
#include "Module/GLSurfaceVisualModule.h"

//IO
#include "GltfFunc.h"
#include "helpers/tinyobj_helper.h"
#include "Field/VehicleInfo.inl"
#include <fstream>

//topo
#include "Mapping/DiscreteElementsToTriangleSet.h"

namespace dyno
{
	//ConfigurableVehicle
	IMPLEMENT_TCLASS(ConfigurableBody, TDataType)

		template<typename TDataType>
	ConfigurableBody<TDataType>::ConfigurableBody()
		: ParametricModel<TDataType>()
		, ArticulatedBody<TDataType>()
	{
		auto elementQuery = std::make_shared<NeighborElementQuery<TDataType>>();
		elementQuery->varSelfCollision()->setValue(false);
		this->stateTopology()->connect(elementQuery->inDiscreteElements());
		this->stateCollisionMask()->connect(elementQuery->inCollisionMask());
		this->stateAttribute()->connect(elementQuery->inAttribute());
		this->animationPipeline()->pushModule(elementQuery);

		auto cdBV = std::make_shared<CollistionDetectionTriangleSet<TDataType>>();
		this->stateTopology()->connect(cdBV->inDiscreteElements());
		this->inTriangleSet()->connect(cdBV->inTriangleSet());
		// 		auto cdBV = std::make_shared<CollistionDetectionBoundingBox<TDataType>>();
		// 		this->stateTopology()->connect(cdBV->inDiscreteElements());
		this->animationPipeline()->pushModule(cdBV);


		auto merge = std::make_shared<ContactsUnion<TDataType>>();
		elementQuery->outContacts()->connect(merge->inContactsA());
		cdBV->outContacts()->connect(merge->inContactsB());
		this->animationPipeline()->pushModule(merge);

		auto iterSolver = std::make_shared<TJConstraintSolver<TDataType>>();
		this->stateTimeStep()->connect(iterSolver->inTimeStep());
		this->varFrictionEnabled()->quote(iterSolver->varFrictionEnabled());
		this->varGravityEnabled()->quote(iterSolver->varGravityEnabled());
		this->varGravityValue()->quote(iterSolver->varGravityValue());
		//this->varFrictionCoefficient()->connect(iterSolver->varFrictionCoefficient());
		this->varFrictionCoefficient()->setValue(20.0f);
		this->varSlop()->quote(iterSolver->varSlop());
		this->stateMass()->connect(iterSolver->inMass());
		this->stateCenter()->connect(iterSolver->inCenter());
		this->stateVelocity()->connect(iterSolver->inVelocity());
		this->stateAngularVelocity()->connect(iterSolver->inAngularVelocity());
		this->stateRotationMatrix()->connect(iterSolver->inRotationMatrix());
		this->stateInertia()->connect(iterSolver->inInertia());
		this->stateQuaternion()->connect(iterSolver->inQuaternion());
		this->stateInitialInertia()->connect(iterSolver->inInitialInertia());

		this->stateTopology()->connect(iterSolver->inDiscreteElements());

		merge->outContacts()->connect(iterSolver->inContacts());

		this->animationPipeline()->pushModule(iterSolver);

		this->inTriangleSet()->tagOptional(true);
		this->varFilePath()->tagOptional(true);
		this->inTextureMesh()->tagOptional(true);

		auto saveCallback = std::make_shared<FCallBackFunc>(std::bind(&ConfigurableBody<TDataType>::saveToFile, this));
		this->varSaveConfigPath()->attach(saveCallback);

		auto loadCallback = std::make_shared<FCallBackFunc>(std::bind(&ConfigurableBody<TDataType>::loadFromFile, this));
		this->varLoadConfigPath()->attach(loadCallback);


		auto updateCallback = std::make_shared<FCallBackFunc>(std::bind(&ConfigurableBody<TDataType>::updateConfig, this));
		this->inTextureMesh()->attach(updateCallback);
		this->varFilePath()->attach(updateCallback);
		this->varConfiguration()->attach(updateCallback);

		auto triElement = std::make_shared<DiscreteElementsToTriangleSet<TDataType>>();
		this->stateTopology()->connect(triElement->inDiscreteElements());
		auto surfaceRender = std::make_shared<GLSurfaceVisualModule>();
		triElement->outTriangleSet()->connect(surfaceRender->inTriangleSet());
		surfaceRender->varAlpha()->setValue(0.5);
		this->graphicsPipeline()->pushModule(triElement);
		this->graphicsPipeline()->pushModule(surfaceRender);

	}

	template<typename TDataType>
	ConfigurableBody<TDataType>::~ConfigurableBody()
	{

	}

	template<typename TDataType>
	void ConfigurableBody<TDataType>::saveToFile()
	{
		/*auto fileStr = this->varFilePath()->serialize();
		auto configStr = this->varConfiguration()->serialize();
		auto instanceTransformStr = this->varVehiclesTransform()->serialize();

		auto Path = this->varSaveConfigPath()->getValue();

		std::ofstream outFile(Path.string(), std::ios::out | std::ios::trunc);
		if (!outFile.is_open())
		{
			throw std::runtime_error("Error Path : " + Path.string());
		}
		MultiBodyBind vehicleBind = getMultiBodyBind();

		outFile << "TextureMesh File:\n" << fileStr << "\n\n";
		outFile << "Configuration:\n" << configStr << "\n\n";
		outFile << "VehiclesTransform:\n" << instanceTransformStr << "\n";

		outFile.close();*/
	}

	template<typename TDataType>
	void ConfigurableBody<TDataType>::loadFromFile()
	{
		//auto Path = this->varLoadConfigPath()->getValue();

		//std::ifstream inFile(Path.string(), std::ios::in);
		//if (!inFile.is_open())
		//{
		//	//throw std::runtime_error("Error Path : " + Path.string());
		//	return;
		//}

		//std::stringstream buffer;
		//buffer << inFile.rdbuf();
		//std::string content = buffer.str();
		//inFile.close();


		//auto extractSection = [](const std::string& text, const std::string& sectionName) -> std::string {
		//	std::string startTag = sectionName + ":";
		//	size_t startPos = text.find(startTag);
		//	if (startPos == std::string::npos)
		//		throw std::runtime_error("Error Section: " + sectionName);

		//	size_t lineEnd = text.find('\n', startPos);
		//	if (lineEnd == std::string::npos)
		//		lineEnd = text.length();

		//	size_t contentStart = lineEnd + 1;

		//	size_t endPos = text.find("\n\n", contentStart);
		//	if (endPos == std::string::npos)
		//		endPos = text.length();

		//	return text.substr(contentStart, endPos - contentStart);
		//};

		//std::string fileStr = extractSection(content, "TextureMesh File");
		//std::string configStr = extractSection(content, "Configuration");
		//std::string instanceTransformStr = extractSection(content, "VehiclesTransform");

		//this->inTextureMesh()->deserialize(fileStr);
		//this->varConfiguration()->deserialize(configStr);
		//this->varVehiclesTransform()->deserialize(instanceTransformStr);
	}

	ElementType ToElementType(ConfigShapeType configShape)
	{
		switch (configShape)
		{
		case CONFIG_BOX:       return ET_BOX;
		case CONFIG_TET:       return ET_TET;
		case CONFIG_CAPSULE:   return ET_CAPSULE;
		case CONFIG_SPHERE:    return ET_SPHERE;
		case CONFIG_TRI:       return ET_TRI;
		case CONFIG_COMPOUND:  return ET_COMPOUND;
		case CONFIG_Other:     return ET_Other;
		default:               return ET_Other; 
		}
	}

	CollisionMask ToCollisionMask(int configMask)
	{
		switch (configMask)
		{
		case RIGID_AllObjects:      return CT_AllObjects;
		case RIGID_BoxExcluded:     return CT_BoxExcluded;
		case RIGID_TetExcluded:     return CT_TetExcluded;
		case RIGID_CapsuleExcluded: return CT_CapsuleExcluded;
		case RIGID_SphereExcluded:  return CT_SphereExcluded;
		case RIGID_BoxOnly:         return CT_BoxOnly;
		case RIGID_TetOnly:         return CT_TetOnly;
		case RIGID_CapsuleOnly:     return CT_CapsuleOnly;
		case RIGID_SphereOnly:      return CT_SphereOnly;
		case RIGID_Disabled:        return CT_Disabled;
		default:                    return CT_AllObjects; 
		}
	}

	BodyType ToBodyType(int configMotion)
	{
		switch (configMotion)
		{
		case RIGID_Static:        return Static;
		case RIGID_Kinematic:     return Kinematic;
		case RIGID_Dynamic:       return Dynamic;
		case RIGID_NonRotatable:  return NonRotatable;
		case RIGID_NonGravitative:return NonGravitative;
		default:                   return Dynamic; 
		}
	}

	template<typename TDataType>
	void ConfigurableBody<TDataType>::updateConfig()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		if (!this->inTextureMesh()->isEmpty())
		{
			this->stateTextureMesh()->setDataPtr(this->inTextureMesh()->constDataPtr());
		}
		else if (this->stateTextureMesh()->isEmpty()) 
		{
			ArticulatedBody<TDataType>::varChanged();
		}
		std::cout << this->varConfiguration()->getValue().varRigidBodyConfigs()->size()<<"\n";
		if (!this->varConfiguration()->getValue().isValid() || !bool(this->varVehiclesTransform()->getValue().size()) || this->stateTextureMesh()->isEmpty())
			return;

		auto texMesh = this->stateTextureMesh()->constDataPtr();
		auto& config = this->varConfiguration()->getValue();

		auto rigidInfo = config.varRigidBodyConfigs();
		std::cout << rigidInfo;

		for (auto it = rigidInfo->begin(); it != rigidInfo->end(); it++)
		{
			auto rigid = rigidInfo->getElement(it);
			auto shapeName = rigid.varShapeName()->getValue();
			std::cout << "ShapeName: " << shapeName <<"\n";
			auto rigidBodyID = rigid.varRigidBodyId()->getValue();
			std::cout << "RigidBodyID: " << rigidBodyID <<"\n";
			auto angle = rigid.varAngel()->getValue();
			std::cout << "Angle: " << angle.x << ", " << angle.y << ", " << angle.z << ", " << angle.w << "\n";
			auto linearVelocity = rigid.varLinearVelocity()->getValue();
			std::cout << "linearVelocity: " << linearVelocity.x << ", " << linearVelocity.y << ", " << linearVelocity.z << "\n";
			auto angularVelocity = rigid.varAngularVelocity()->getValue();
			std::cout << "angularVelocity: " << angularVelocity.x << ", " << angularVelocity.y << ", " << angularVelocity.z << "\n";
			auto position = rigid.varPosition()->getValue();
			std::cout << "position: " << position.x << ", " << position.y << ", " << position.z << "\n";
			auto offset = rigid.varOffset()->getValue();
			std::cout << "offset: " << offset.x << ", " << offset.y << ", " << offset.z << "\n";
			auto inertia = rigid.varInertia()->getValue();
			std::cout << "inertia: " << inertia(0, 0) << ", " << inertia(0, 1) << ", " << inertia(0, 2) << "\n" << inertia(1, 0) << ", " << inertia(1, 1) << ", " << inertia(1, 2) << "\n" << inertia(2, 0) << ", " << inertia(2, 1) << ", " << inertia(2, 2) << "\n";
			auto friction = rigid.varFriction()->getValue();
			std::cout << "friction: " << friction << "\n";
			auto restitution = rigid.varRestitution()->getValue();
			std::cout << "restitution: " << restitution << "\n";
			auto motionType = rigid.varMotionType()->getValue();
			std::cout << "motionType: " << motionType.currentString() << "\n";
			auto shapeType = rigid.varShapeType()->getValue();
			std::cout << "shapeType: " << shapeType.currentString() << "\n";
			auto collisionMask = rigid.varCollisionMask()->getValue();
			std::cout << "collisionMask: " << collisionMask.currentString() << "\n";
			auto ConfigGroup = rigid.varConfigGroup()->getValue();
			std::cout << "ConfigGroup: " << ConfigGroup << "\n";

			auto visualShapeIds = rigid.varVisualShapeIds();
			for (auto visualIdIterator = visualShapeIds->begin(); visualIdIterator != visualShapeIds->end(); visualIdIterator++)
			{
				auto visualID = visualShapeIds->getElement(visualIdIterator);
				std::cout << "    visualShapeIds: " << visualID << "\n";

			}

			auto shapeConfigs = rigid.varShapeConfigs();
			for (auto shapeIterator = shapeConfigs->begin(); shapeIterator != shapeConfigs->end(); shapeIterator++) 
			{
				auto shape = shapeConfigs->getElement(shapeIterator);

				std::cout << "		 shapeConfigs: " << "\n";
				std::cout << "       capsuleLength: " << shape.varCapsuleLength()->getValue() << "\n";
				auto center = shape.varCenter()->getValue();
				std::cout << "       center: " << center.x << ", " << center.y << ", " << center.z << "\n";
				std::cout << "       density: " << shape.varDensity()->getValue() << "\n";
				auto halfLength = shape.varHalfLength()->getValue();
				std::cout << "       halfLength: " << halfLength.x << "," << halfLength.y << "," << halfLength.z << "\n";
				std::cout << "       radius: " << shape.varRadius()->getValue() << "\n";
				auto rot = shape.varRot()->getValue();
				std::cout << "       rot: " << rot.x << "," << rot.y << "," << rot.z << "," << rot.w << "\n";
				auto shapeType = shape.varShapeType()->getValue();
				std::cout << "       shapeType: " << shapeType.currentString() << "\n";
			}

		}

		const auto jointInfo = config.varJointConfigs();
		for (auto jointIterator = jointInfo->begin(); jointIterator != jointInfo->end(); jointIterator++)
		{
			auto joint = jointInfo->getElement(jointIterator);

			std::cout << "mJointType: " << joint.varJointType()->getValue().currentString() << "\n";
			std::cout << "ARigidBodyName: " << joint.varAShapeName()->getValue() << "\n";
			std::cout << "BRigidBodyName: " << joint.varBShapeName()->getValue() << "\n";
			auto anchorPoint = joint.varAnchorPoint()->getValue();
			std::cout << "mAnchorPoint: " << anchorPoint.x << "," << anchorPoint.y << "," << anchorPoint.z << "\n";

			std::cout << "mUseMoter: " << joint.varUseMoter()->getValue() << "\n";
			std::cout << "mUseRange: " << joint.varUseRange()->getValue() << "\n";
			std::cout << "mMax: " << joint.varRange()->getValue().y << "\n";
			std::cout << "mMin: " << joint.varRange()->getValue().x << "\n";
			std::cout << "mMoter: " << joint.varMoter()->getValue() << "\n";
			auto axis = joint.varAxis()->getValue();
			std::cout << "mAxis: " << axis.x << "," << axis.y << "," << axis.z << "\n";
			auto q = joint.varQ()->getValue();
			std::cout << "q: " << q.x << "," << q.y << "," << q.z << ","<< q.w << "\n";
			auto r1 = joint.varR1()->getValue();
			auto r2 = joint.varR2()->getValue();
			std::cout << "r1: " << r1.x << "," << r1.y << "," << r1.z << "\n";
			std::cout << "r2: " << r2.x << "," << r2.y << "," << r2.z << "\n";
			auto distance = joint.varDistance()->getValue();
			std::cout << "distance: " << distance << "\n";
		}

		// **************************** Create RigidBody  **************************** //
		auto instances = this->varVehiclesTransform()->getValue();
		uint vehicleNum = instances.size();
		int maxGroup = 0;

		for (auto rigid = rigidInfo->begin(); rigid != rigidInfo->end(); rigid++)
		{
			auto configGroup = rigidInfo->getElement(rigid).varConfigGroup()->getValue();
			if (configGroup > maxGroup)
				maxGroup = configGroup;
		}


		for (size_t j = 0; j < vehicleNum; j++)
		{
			std::vector<std::shared_ptr<PdActor>> Actors;
			Actors.resize(rigidInfo->size());
			int i = -1;
			for (auto rigidIterator = rigidInfo->begin(); rigidIterator != rigidInfo->end(); rigidIterator++)
			{
				i++;
				auto rigid = rigidInfo->getElement(rigidIterator);
				int visualId = -1;
				std::shared_ptr<Shape> visualShapePtr = NULL;
				if (rigid.varVisualShapeIds()->size())
				{
					int validIndex = int(texMesh->shapes().size()) - 1;
								
					if(rigid.varVisualShapeIds()->size())
						visualId = rigid.varVisualShapeIds()->getElement(rigid.varVisualShapeIds()->begin());

					if (visualId <= validIndex && visualId >= 0)
					{
						visualShapePtr = texMesh->shapes()[visualId];
					}
					else
					{
						visualId = texMesh->shapes().size() - 1;
					}
				}
				
				RigidBodyInfo rigidbody;

				rigidbody.bodyId = j * maxGroup + rigid.varConfigGroup()->getValue();
				rigidbody.angle = rigid.varAngel()->getValue() * Quat<Real>(instances[j].rotation());
				rigidbody.linearVelocity = rigid.varLinearVelocity()->getValue();
				rigidbody.angularVelocity = rigid.varAngularVelocity()->getValue();
				
				if (!visualShapePtr) 
				{
					if (std::isnan(rigidbody.position.x))
					{
						rigidbody.position = Vec3f(0);
					}
					else
						rigidbody.position = rigid.varPosition()->getValue();
				}
				else 
				{
					if (std::isnan(rigidbody.position.x))
					{
						rigidbody.position = visualShapePtr->boundingTransform.translation();
					}
					else 
					{
						rigidbody.position = visualShapePtr->boundingTransform.translation() + rigidbody.position;
					}
				}
				
				rigidbody.position = Quat<Real>(instances[j].rotation()).rotate(rigidbody.position) + instances[j].translation();

				rigidbody.offset = rigid.varOffset()->getValue();
				rigidbody.inertia = rigid.varInertia()->getValue();
				rigidbody.friction = rigid.varFriction()->getValue() == -1 ? this->varFrictionCoefficient()->getValue() : rigid.varFriction()->getValue();
				rigidbody.restitution = rigid.varRestitution()->getValue();
				rigidbody.motionType = ToBodyType(rigid.varMotionType()->currentKey());
				rigidbody.shapeType = ElementType::ET_COMPOUND;
				rigidbody.collisionMask = ToCollisionMask(rigid.varCollisionMask()->currentKey());
				
				Actors[i] = this->createRigidBody(rigidbody);

				for (auto elementIterator = rigid.varShapeConfigs()->begin(); elementIterator != rigid.varShapeConfigs()->end(); elementIterator++)
				{
					auto element = rigid.varShapeConfigs()->getElement(elementIterator);
					Vec3f up;
					Vec3f down;
					Vec3f T;

					if (visualShapePtr)
					{
						up = visualShapePtr->boundingBox.v1;
						down = visualShapePtr->boundingBox.v0;
						T = visualShapePtr->boundingTransform.translation();
					}

					switch (element.varShapeType()->currentKey())
					{
						case SHAPE_BOX:
						{
							BoxInfo currentBox;
							currentBox.center = element.varCenter()->getValue();
							currentBox.rot = element.varRot()->getValue();
							if (element.varHalfLength()->getValue() == Vector<Real, 3>(0) && visualShapePtr)
								currentBox.halfLength = (up - down) / 2;
							else
								currentBox.halfLength = element.varHalfLength()->getValue();

							this->bindBox(Actors[i], currentBox, element.varDensity()->getValue());
							break;
						}

						break;
						case SHAPE_TET: 
						{
							TetInfo currentTet;
							float Length = 0;
							for (auto tetIterator = element.varTet()->begin(); tetIterator != element.varTet()->end(); tetIterator++)
							{
								auto tetCord = element.varTet()->getElement(tetIterator);
								Length += tetCord.norm();
							}	

							if(Length == 0 )
							{
								std::vector<Vector<Real, 3>> v[4];
								currentTet.v[0] = (visualShapePtr ? down : Vec3f(0));
								currentTet.v[1] = (visualShapePtr ? Vec3f(up.x, down.y, down.z) : Vec3f(1, 0, 0));
								currentTet.v[2] = (visualShapePtr ? Vec3f(up.x, down.y, up.z) : Vec3f(0, 1, 0));
								currentTet.v[3] = (visualShapePtr ? Vec3f(up) : Vec3f(0, 0, 1));
							}
							else if(Length > 0)
							{
								int tetId = 0;
								for (auto tetIterator = element.varTet()->begin(); tetIterator != element.varTet()->end(); tetIterator++)
								{
									auto tetCoord = element.varTet()->getElement(tetIterator);
									currentTet.v[tetId] = tetCoord;
									tetId++;
								}
							}
							this->bindTet(Actors[i], currentTet, element.varDensity()->getValue());

							break;
						}
						case SHAPE_CAPSULE:
						{
							CapsuleInfo currentCapsule;
							currentCapsule.center = element.varCenter()->getValue();
							currentCapsule.rot = element.varRot()->getValue();

							if (element.varCapsuleLength()->getValue() != 0 && element.varRadius()->getValue() != 0)
							{
								currentCapsule.halfLength = element.varCapsuleLength()->getValue();
								currentCapsule.radius = element.varRadius()->getValue();
							}
							else 
							{
								if (element.varCapsuleLength()->getValue() == 0 && visualShapePtr)
									currentCapsule.halfLength = (up.y - down.y) / 2;
								else if(element.varCapsuleLength()->getValue() != 0)
									currentCapsule.halfLength = element.varCapsuleLength()->getValue();

								if (element.varRadius()->getValue() == 0 && visualShapePtr)
									currentCapsule.radius = std::abs(up.y - down.y) / 2;
								else if (element.varRadius()->getValue() != 0)
									currentCapsule.radius = element.varRadius()->getValue();
							}

							this->bindCapsule(Actors[i], currentCapsule, element.varDensity()->getValue());
							break;
						}
						case SHAPE_SPHERE: 
						{
							SphereInfo currentSphere;
							currentSphere.center = element.varCenter()->getValue();
							currentSphere.rot = element.varRot()->getValue();
							if (element.varRadius()->getValue() == 0 && visualShapePtr)
							{
								currentSphere.radius = std::abs(up.y - down.y) / 2;
							}
							else 
							{
								currentSphere.radius = element.varRadius()->getValue();
							}
							this->bindSphere(Actors[i], currentSphere, element.varDensity()->getValue());
							break;
						}
						default:
							break;
					}
				}

				if (visualId!= -1 && Actors[i] != NULL)
				{
					////bindShapetoActor
					this->bindShape(Actors[i], Pair<uint, uint>(visualId, j));
				}
			}

			auto jointInfo = config.varJointConfigs();
			for (auto jointIterator = jointInfo->begin(); jointIterator != jointInfo->end(); jointIterator++)
			{
				auto jointDetail = jointInfo->getElement(jointIterator);
				//Actor
				auto type = jointDetail.varJointType()->currentKey();
				int first = jointDetail.varARigidBodyId()->getValue();
				int second = jointDetail.varBRigidBodyId()->getValue();
				Real speed = jointDetail.varMoter()->getValue();
				auto axis = Quat1f(instances[j].rotation()).rotate(jointDetail.varAxis()->getValue());
				auto anchorOffset = jointDetail.varAnchorPoint()->getValue();

				if (first == -1 || second == -1)
				{
					printf("JointInfo : id == -1 [%d], [%d]\n", first, second);
					continue;
				}
				if (first >= Actors.size() || second >= Actors.size())
				{
					printf("JointInfo : Error RigidId  [%d], [%d]\n", first, second);
					continue;
				}
				if (Actors[first] == NULL || Actors[second] == NULL) 
				{
					printf("JointInfo : Actor is NULL [%d], [%d]\n", first, second);
					continue;
				}


				if (type == CONFIG_Hinge)
				{
					auto& hingeJoint = this->createHingeJoint(Actors[first], Actors[second]);
					hingeJoint.setAnchorPoint(Actors[first]->center + anchorOffset);
					hingeJoint.setAxis(axis);
					if (jointDetail.varUseMoter()->getValue())
						hingeJoint.setMoter(speed);
					if (jointDetail.varUseRange()->getValue())
						hingeJoint.setRange(jointDetail.varRange()->getValue().x, jointDetail.varRange()->getValue().y);

				}
				if (type == CONFIG_Slider)
				{
					auto& sliderJoint = this->createSliderJoint(Actors[first], Actors[second]);
					sliderJoint.setAnchorPoint((Actors[first]->center + Actors[first]->center) / 2 + anchorOffset);
					sliderJoint.setAxis(axis);
					if (jointDetail.varUseMoter()->getValue())
						sliderJoint.setMoter(speed);
					if (jointDetail.varUseRange()->getValue())
						sliderJoint.setRange(jointDetail.varRange()->getValue().x, jointDetail.varRange()->getValue().y);
				}
				if (type == CONFIG_Fixed)
				{
					auto& fixedJoint1 = this->createFixedJoint(Actors[first], Actors[second]);
					fixedJoint1.setAnchorPoint((Actors[first]->center + Actors[first]->center) / 2 + anchorOffset);
				}
				if (type == CONFIG_Point)
				{
					auto& pointJoint = this->createPointJoint(Actors[first]);
					pointJoint.setAnchorPoint(Actors[first]->center + anchorOffset);
				}
				if (type == CONFIG_BallAndSocket)
				{
					auto& ballAndSocketJoint = this->createBallAndSocketJoint(Actors[first], Actors[second]);
					ballAndSocketJoint.setAnchorPoint((Actors[first]->center + Actors[first]->center) / 2 + anchorOffset);
				}
			}
		}

	}

	template<typename TDataType>
	void ConfigurableBody<TDataType>::resetStates()
	{
		/***************** Reset *************/
		//loadFromFile();
		updateConfig();

		ArticulatedBody<TDataType>::resetStates();

		RigidBodySystem<TDataType>::postUpdateStates();

		this->updateInstanceTransform();
	}

	DEFINE_CLASS(ConfigurableBody);



}
