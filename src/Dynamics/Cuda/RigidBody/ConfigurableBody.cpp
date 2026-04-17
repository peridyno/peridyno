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

//IO
#include "GltfFunc.h"
#include "helpers/tinyobj_helper.h"
#include "Field/VehicleInfo.inl"
#include <fstream>

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


	}

	template<typename TDataType>
	ConfigurableBody<TDataType>::~ConfigurableBody()
	{

	}

	template<typename TDataType>
	void ConfigurableBody<TDataType>::saveToFile()
	{
		auto fileStr = this->varFilePath()->serialize();
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

		outFile.close();
	}

	template<typename TDataType>
	void ConfigurableBody<TDataType>::loadFromFile()
	{
		auto Path = this->varLoadConfigPath()->getValue();

		std::ifstream inFile(Path.string(), std::ios::in);
		if (!inFile.is_open())
		{
			//throw std::runtime_error("Error Path : " + Path.string());
			return;
		}

		std::stringstream buffer;
		buffer << inFile.rdbuf();
		std::string content = buffer.str();
		inFile.close();


		auto extractSection = [](const std::string& text, const std::string& sectionName) -> std::string {
			std::string startTag = sectionName + ":";
			size_t startPos = text.find(startTag);
			if (startPos == std::string::npos)
				throw std::runtime_error("Error Section: " + sectionName);

			size_t lineEnd = text.find('\n', startPos);
			if (lineEnd == std::string::npos)
				lineEnd = text.length();

			size_t contentStart = lineEnd + 1;

			size_t endPos = text.find("\n\n", contentStart);
			if (endPos == std::string::npos)
				endPos = text.length();

			return text.substr(contentStart, endPos - contentStart);
		};

		std::string fileStr = extractSection(content, "TextureMesh File");
		std::string configStr = extractSection(content, "Configuration");
		std::string instanceTransformStr = extractSection(content, "VehiclesTransform");

		this->inTextureMesh()->deserialize(fileStr);
		this->varConfiguration()->deserialize(configStr);
		this->varVehiclesTransform()->deserialize(instanceTransformStr);
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
		default:               return ET_Other; // 默认返回
		}
	}

	CollisionMask ToCollisionMask(ConfigCollisionMask configMask)
	{
		switch (configMask)
		{
		case CONFIG_AllObjects:      return CT_AllObjects;
		case CONFIG_BoxExcluded:     return CT_BoxExcluded;
		case CONFIG_TetExcluded:     return CT_TetExcluded;
		case CONFIG_CapsuleExcluded: return CT_CapsuleExcluded;
		case CONFIG_SphereExcluded:  return CT_SphereExcluded;
		case CONFIG_BoxOnly:         return CT_BoxOnly;
		case CONFIG_TetOnly:         return CT_TetOnly;
		case CONFIG_CapsuleOnly:     return CT_CapsuleOnly;
		case CONFIG_SphereOnly:      return CT_SphereOnly;
		case CONFIG_Disabled:        return CT_Disabled;
		default:                    return CT_AllObjects; // 默认返回
		}
	}

	BodyType ToBodyType(ConfigMotionType configMotion)
	{
		switch (configMotion)
		{
		case CONFIG_Static:        return Static;
		case CONFIG_Kinematic:     return Kinematic;
		case CONFIG_Dynamic:       return Dynamic;
		case CONFIG_NonRotatable:  return NonRotatable;
		case CONFIG_NonGravitative:return NonGravitative;
		default:                   return Dynamic; // 默认返回
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

		if (!this->varConfiguration()->getValue().isValid() && !bool(this->varVehiclesTransform()->getValue().size()) || this->stateTextureMesh()->isEmpty())
			return;

		auto texMesh = this->stateTextureMesh()->constDataPtr();
		const auto config = this->varConfiguration()->getValue();

		const auto rigidInfo = config.rigidBodyConfigs;
		const auto jointInfo = config.jointConfigs;

		for (size_t i = 0; i < rigidInfo.size(); i++)
		{
			std::cout << "ShapeName: " << rigidInfo[i].shapeName.name << "  -  " << rigidInfo[i].shapeName.rigidBodyId << "\n";
			std::cout << "Angle: " << rigidInfo[i].angle.x << ", " << rigidInfo[i].angle.y << ", " << rigidInfo[i].angle.z << ", " << rigidInfo[i].angle.w << "\n";
			std::cout << "linearVelocity: " << rigidInfo[i].linearVelocity.x << ", " << rigidInfo[i].linearVelocity.y << ", " << rigidInfo[i].linearVelocity.z  << "\n";
			std::cout << "angularVelocity: " << rigidInfo[i].angularVelocity.x << ", " << rigidInfo[i].angularVelocity.y << ", " << rigidInfo[i].angularVelocity.z << "\n";
			std::cout << "position: " << rigidInfo[i].position.x << ", " << rigidInfo[i].position.y << ", " << rigidInfo[i].position.z << "\n";
			std::cout << "offset: " << rigidInfo[i].offset.x << ", " << rigidInfo[i].offset.y << ", " << rigidInfo[i].offset.z << "\n";
			std::cout << "inertia: " << rigidInfo[i].inertia(0, 0) << ", " << rigidInfo[i].inertia(0, 1) << ", " << rigidInfo[i].inertia(0, 2) << "\n" << rigidInfo[i].inertia(1, 0) << ", " << rigidInfo[i].inertia(1, 1) << ", " << rigidInfo[i].inertia(1, 2) << "\n" << rigidInfo[i].inertia(2, 0) << ", " << rigidInfo[i].inertia(2, 1) << ", " << rigidInfo[i].inertia(2, 2) << "\n";
			std::cout << "friction: " << rigidInfo[i].friction << "\n";
			std::cout << "restitution: " << rigidInfo[i].restitution << "\n";
			std::cout << "motionType: " << rigidInfo[i].motionType << "\n";
			std::cout << "shapeType: " << rigidInfo[i].shapeType << "\n";
			std::cout << "collisionMask: " << rigidInfo[i].collisionMask << "\n";

			std::cout << "ConfigGroup: " << rigidInfo[i].ConfigGroup << "\n";
			for (size_t j = 0; j < rigidInfo[i].visualShapeIds.size(); j++)
			{
				std::cout << "    visualShapeIds: " << rigidInfo[i].visualShapeIds[j] << "\n";
			}
			for (size_t j = 0; j < rigidInfo[i].shapeConfigs.size(); j++)
			{
				std::cout << "    shapeConfigs: " << "\n";
				std::cout << "       capsuleLength: " << rigidInfo[i].shapeConfigs[j].capsuleLength << "\n";
				std::cout << "       center: " << rigidInfo[i].shapeConfigs[j].center.x << ", " << rigidInfo[i].shapeConfigs[j].center.y << ", " << rigidInfo[i].shapeConfigs[j].center.z << "\n";
				std::cout << "       density: " << rigidInfo[i].shapeConfigs[j].density << "\n";
				std::cout << "       halfLength: " << rigidInfo[i].shapeConfigs[j].halfLength.x <<"," << rigidInfo[i].shapeConfigs[j].halfLength.y << "," << rigidInfo[i].shapeConfigs[j].halfLength.z << "\n";
				std::cout << "       radius: " << rigidInfo[i].shapeConfigs[j].radius << "\n";
				std::cout << "       rot: " << rigidInfo[i].shapeConfigs[j].rot.x << "," << rigidInfo[i].shapeConfigs[j].rot.y << "," << rigidInfo[i].shapeConfigs[j].rot.z << "," << rigidInfo[i].shapeConfigs[j].rot.w << "\n";
				std::cout << "       shapeType: " << rigidInfo[i].shapeConfigs[j].shapeType << "\n";
				//std::cout << "       shapeConfigs: " << rigidInfo[i].shapeConfigs[j].tet << "\n";
			}
		}

		for (size_t i = 0; i < jointInfo.size(); i++)
		{
			std::cout << "mJointType: " << jointInfo[i].mJointType << "\n";
			std::cout << "mRigidBodyName_1: " << jointInfo[i].mRigidBodyName_1.name << "\n";
			std::cout << "mRigidBodyName_2: " << jointInfo[i].mRigidBodyName_2.name << "\n";
			std::cout << "mAnchorPoint: " << jointInfo[i].mAnchorPoint.x << "," << jointInfo[i].mAnchorPoint.y << "," << jointInfo[i].mAnchorPoint.z << "\n";
			std::cout << "mUseMoter: " << jointInfo[i].mUseMoter << "\n";
			std::cout << "mUseRange: " << jointInfo[i].mUseRange << "\n";
			std::cout << "mMax: " << jointInfo[i].mMax << "\n";
			std::cout << "mMin: " << jointInfo[i].mMin << "\n";
			std::cout << "mMoter: " << jointInfo[i].mMoter << "\n";
			std::cout << "mAxis: " << jointInfo[i].mAxis.x << "," << jointInfo[i].mAxis.y << "," << jointInfo[i].mAxis.z << "\n";
			std::cout << "q: " << jointInfo[i].q.x << "," << jointInfo[i].q.y << "," << jointInfo[i].q.z << ","<< jointInfo[i].q.w << "\n";
			std::cout << "r1: " << jointInfo[i].r1.x << "," << jointInfo[i].r1.y << "," << jointInfo[i].r1.z << "\n";
			std::cout << "r2: " << jointInfo[i].r2.x << "," << jointInfo[i].r2.y << "," << jointInfo[i].r2.z << "\n";
			std::cout << "distance: " << jointInfo[i].distance << "\n";
		}

		// **************************** Create RigidBody  **************************** //
		auto instances = this->varVehiclesTransform()->getValue();
		uint vehicleNum = instances.size();
		int maxGroup = 0;
		for (size_t i = 0; i < rigidInfo.size(); i++)
		{
			if (rigidInfo[i].ConfigGroup > maxGroup)
				maxGroup = rigidInfo[i].ConfigGroup;
		}

		for (size_t j = 0; j < vehicleNum; j++)
		{


			std::vector<std::shared_ptr<PdActor>> Actors;

			Actors.resize(rigidInfo.size());


			for (size_t i = 0; i < rigidInfo.size(); i++)
			{
				std::shared_ptr<Shape> visualShapePtr = NULL;
				int visualID = -1;
				if (rigidInfo[i].visualShapeIds.size())
				{
					int validIndex = int(texMesh->shapes().size()) - 1;
					
					if (rigidInfo[i].visualShapeIds[0] <= validIndex && rigidInfo[i].visualShapeIds[0] >= 0)
					{
						visualID = rigidInfo[i].visualShapeIds[0];
						visualShapePtr = texMesh->shapes()[visualID];
					}
					else
					{
						visualID = texMesh->shapes().size() - 1;
					}
				}

				RigidBodyInfo rigidbody;

				rigidbody.bodyId = j * maxGroup + rigidInfo[i].ConfigGroup;
				rigidbody.angle = rigidInfo[i].angle * Quat<Real>(instances[j].rotation());
				rigidbody.linearVelocity = rigidInfo[i].linearVelocity;
				rigidbody.angularVelocity = rigidInfo[i].angularVelocity;
				if (!visualShapePtr) 
				{
					if (std::isnan(rigidbody.position.x))
					{
						rigidbody.position = Vec3f(0);
					}
					else
						rigidbody.position = rigidInfo[i].position;
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

				rigidbody.offset = rigidInfo[i].offset;
				rigidbody.inertia = rigidInfo[i].inertia;
				rigidbody.friction = rigidInfo[i].friction == -1 ? this->varFrictionCoefficient()->getValue() : rigidInfo[i].friction;
				rigidbody.restitution = rigidInfo[i].restitution;
				rigidbody.motionType = ToBodyType(rigidInfo[i].motionType);
				rigidbody.shapeType = ElementType::ET_COMPOUND;
				rigidbody.collisionMask = ToCollisionMask(rigidInfo[i].collisionMask);

				Actors[i] = this->createRigidBody(rigidbody);

				

				for (size_t elementID = 0; elementID < rigidInfo[i].shapeConfigs.size(); elementID++)
				{
					auto element = rigidInfo[i].shapeConfigs[elementID];
					Vec3f up;
					Vec3f down;
					Vec3f T;

					if (visualShapePtr)
					{
						up = visualShapePtr->boundingBox.v1;
						down = visualShapePtr->boundingBox.v0;
						T = visualShapePtr->boundingTransform.translation();
					}

					switch (element.shapeType)
					{
					case CONFIG_BOX:
					{
						BoxInfo currentBox;
						currentBox.center = element.center;
						currentBox.rot = element.rot;
						if (element.halfLength == Vector<Real, 3>(0) && visualShapePtr)
							currentBox.halfLength = (up - down) / 2;
						else
							currentBox.halfLength = element.halfLength;

						this->bindBox(Actors[i], currentBox, element.density);
						break;
					}

					break;
					case CONFIG_TET: 
					{
						TetInfo currentTet;
						float Length = 0;
						for (auto tet : element.tet)
							Length += tet.norm(); 


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
							currentTet.v[0] = element.tet[0];
							currentTet.v[1] = element.tet[1];
							currentTet.v[2] = element.tet[2];
							currentTet.v[3] = element.tet[3];
						}
						this->bindTet(Actors[i], currentTet, element.density);

						break;
					}
					case CONFIG_CAPSULE: 
					{
						CapsuleInfo currentCapsule;
						currentCapsule.center = element.center;
						currentCapsule.rot = element.rot;

						if (element.capsuleLength != 0 && element.radius != 0)
						{
							currentCapsule.halfLength = element.capsuleLength;
							currentCapsule.radius = element.radius;
						}
						else 
						{
							if (element.capsuleLength == 0 && visualShapePtr)
								currentCapsule.halfLength = (up.y - down.y) / 2;
							if (element.radius == 0 && visualShapePtr)
								currentCapsule.radius = std::abs(up.y - down.y) / 2;
						}

						this->bindCapsule(Actors[i], currentCapsule, element.density);
						break;
					}
					case CONFIG_SPHERE: 
					{
						SphereInfo currentSphere;
						currentSphere.center = element.center;
						currentSphere.rot = element.rot;
						if (element.radius == 0 && visualShapePtr)
						{
							currentSphere.radius = std::abs(up.y - down.y) / 2;
						}
						else 
						{
							currentSphere.radius = element.radius;
						}
						this->bindSphere(Actors[i], currentSphere, element.density);
						break;
					}
					default:
						break;
					}
				}

				if (visualID != -1 && Actors[i] != NULL)
				{
					////bindShapetoActor
					this->bindShape(Actors[i], Pair<uint, uint>(visualID, j));
				}
			}

				
			for (size_t i = 0; i < jointInfo.size(); i++)
			{
				//Actor
				auto type = jointInfo[i].mJointType;
				int first = jointInfo[i].mRigidBodyName_1.rigidBodyId;
				int second = jointInfo[i].mRigidBodyName_2.rigidBodyId;
				Real speed = jointInfo[i].mMoter;
				auto axis = Quat1f(instances[j].rotation()).rotate(jointInfo[i].mAxis);
				auto anchorOffset = jointInfo[i].mAnchorPoint;

				if (first == -1 || second == -1)
					continue;
				if (Actors[first] == NULL || Actors[second] == NULL)
					continue;


				if (type == CONFIG_Hinge)
				{
					auto& joint = this->createHingeJoint(Actors[first], Actors[second]);
					joint.setAnchorPoint(Actors[first]->center + anchorOffset);
					joint.setAxis(axis);
					if (jointInfo[i].mUseMoter)
						joint.setMoter(speed);
					if (jointInfo[i].mUseRange)
						joint.setRange(jointInfo[i].mMin, jointInfo[i].mMax);

				}
				if (type == CONFIG_Slider)
				{
					auto& sliderJoint = this->createSliderJoint(Actors[first], Actors[second]);
					sliderJoint.setAnchorPoint((Actors[first]->center + Actors[first]->center) / 2 + anchorOffset);
					sliderJoint.setAxis(axis);
					if (jointInfo[i].mUseMoter)
						sliderJoint.setMoter(speed);
					if (jointInfo[i].mUseRange)
						sliderJoint.setRange(jointInfo[i].mMin, jointInfo[i].mMax);
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
