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
#include <fstream>

//topo
#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "MultiBodyTuple.h"
#include "SceneLoaderXML.h"
#include "GLDiscreteElementVisualModule.h"
#include "GLDiscreteElementJointVisualModule.h"

namespace dyno
{
	//ConfigurableVehicle
	IMPLEMENT_TCLASS(ConfigurableBody, TDataType)

	namespace path_helper
	{
		static bool isDataAbsolutePath(const std::string& filePath)
		{
			std::string assetPath = getAssetPath();

			std::string normFile = filePath;
			std::string normAsset = assetPath;

			for (size_t i = 0; i < normFile.size(); i++)
				if (normFile[i] == '\\') normFile[i] = '/';
			for (size_t i = 0; i < normAsset.size(); i++)
				if (normAsset[i] == '\\') normAsset[i] = '/';

			if (!normAsset.empty() && normAsset.back() != '/')
				normAsset += '/';

			if (normFile.size() < normAsset.size())
				return false;

			for (size_t i = 0; i < normAsset.size(); i++)
			{
				char cf = normFile[i];
				char ca = normAsset[i];
				if (cf >= 'A' && cf <= 'Z') cf += 32;
				if (ca >= 'A' && ca <= 'Z') ca += 32;
				if (cf != ca)
					return false;
			}

			return true;
		}

		static std::string getRelativeToDataPath(const std::string& filePath)
		{
			if (!isDataAbsolutePath(filePath))
				return filePath;

			std::string assetPath = getAssetPath();

			std::string normFile = filePath;
			std::string normAsset = assetPath;

			for (size_t i = 0; i < normFile.size(); i++)
				if (normFile[i] == '\\') normFile[i] = '/';
			for (size_t i = 0; i < normAsset.size(); i++)
				if (normAsset[i] == '\\') normAsset[i] = '/';

			if (!normAsset.empty() && normAsset.back() != '/')
				normAsset += '/';

			return normFile.substr(normAsset.size());
		}

	}

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

		auto elementRender = std::make_shared<GLDiscreteElementVisualModule<TDataType>>();
		this->stateTopology()->connect(elementRender->inDiscreteElements());
		this->graphicsPipeline()->pushModule(elementRender);

		auto jointRender = std::make_shared<GLDiscreteElementJointVisualModule<TDataType>>();
		this->stateTopology()->connect(jointRender->inDiscreteElements());
		this->graphicsPipeline()->pushModule(jointRender);

		auto assetCallback = std::make_shared<FCallBackFunc>(std::bind(&ConfigurableBody<TDataType>::onTexMeshLoad, this));
		this->varConfiguration()->getValue().varAssetConfigs()->attach(assetCallback);

	}

	template<typename TDataType>
	ConfigurableBody<TDataType>::~ConfigurableBody()
	{

	}

	template<typename TDataType>
	void ConfigurableBody<TDataType>::saveToPath(std::string path)
	{
		auto fileString = this->varFilePath()->serialize();

		SceneLoaderXML saveHelper;
		tinyxml2::XMLDocument doc;

		tinyxml2::XMLElement* file = doc.NewElement("MeshFile");
		file->SetText(path_helper::getRelativeToDataPath(fileString).c_str());
		doc.InsertFirstChild(file);

		tinyxml2::XMLElement* transform = doc.NewElement("Transform");
		saveHelper.serializeField(this->varVehiclesTransform(), transform, doc);
		doc.InsertFirstChild(transform);

		tinyxml2::XMLElement* config = doc.NewElement("Configuration");
		saveHelper.serializeField(this->varConfiguration(), config, doc);
		doc.InsertFirstChild(config);

		doc.SaveFile(path.c_str());
	}

	template<typename TDataType>
	void ConfigurableBody<TDataType>::saveToFile()
	{
		FilePath path = this->varSaveConfigPath()->getValue();
		saveToPath(path.string());
	}

	template<typename TDataType>
	void ConfigurableBody<TDataType>::loadFromFile()
	{
		auto path = this->varLoadConfigPath()->getValue().string();

		tinyxml2::XMLDocument doc;
		if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS)
		{
			doc.PrintError();
			std::cout << "Error Load" << std::endl;
			return ;
		}

		this->varConfiguration()->setValue(MultiBodyTuple());

		SceneLoaderXML saveHelper;

		std::map<std::string, FBase*> fieldMap;
		auto& params = this->getParameters();
		for (const auto& f : params)
		{
			fieldMap[f->getObjectName()] = f;
			FList* listPtr = dynamic_cast<FList*>(f);
			if (listPtr)
				listPtr->clear();
		}

		tinyxml2::XMLElement* texfileXmls = doc.FirstChildElement("MeshFile");
		if(texfileXmls)
		{
			const char* text = texfileXmls->GetText();
			std::string path = std::string(text);
			path = path_helper::isDataAbsolutePath(path) ? path : getAssetPath() + path;
			this->varFilePath()->setValue(FilePath(path));
		}

		tinyxml2::XMLElement* transXmls = doc.FirstChildElement("Transform");
		if(transXmls)
			saveHelper.deserializeField(transXmls, fieldMap);

		tinyxml2::XMLElement* configXmls = doc.FirstChildElement("Configuration");
		if(configXmls)
			saveHelper.deserializeField(configXmls, fieldMap);

		this->fileChanged();
		this->updateConfig();
	}

	ElementType ToElementType(RigidShapeType configShape)
	{
		switch (configShape)
		{
		case SHAPE_BOX:			     return ET_BOX;
		case SHAPE_TET:				 return ET_TET;
		case SHAPE_CAPSULE:			 return ET_CAPSULE;
		case SHAPE_SPHERE:			 return ET_SPHERE;
		case SHAPE_TRI:				 return ET_TRI;
		case SHAPE_COMPOUND:		 return ET_COMPOUND;
		case SHAPE_MEDIALCONE:		 return ET_MEDIALCONE;
		case SHAPE_MEDIALSLAB:		 return ET_MEDIALSLAB;
		case SHAPE_Other:			 return ET_Other;

		default:					 return ET_Other; 
		}
	}

	CollisionMask ToCollisionMask(unsigned int configMask)
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

		if (this->stateTextureMesh()->isEmpty() && (!this->varFilePath()->getValue().string().empty() || this->varConfiguration()->getValue().varAssetConfigs()->size()))
		{
			ArticulatedBody<TDataType>::fileChanged();
		}

		if(!this->inTextureMesh()->isEmpty() && this->stateTextureMesh()->constDataPtr()->shapes().size() == 0)
			this->stateTextureMesh()->getDataPtr()->assign(this->inTextureMesh()->constDataPtr());

		auto texMesh = this->stateTextureMesh()->constDataPtr();

		if (!this->varConfiguration()->getValue().isValid() || !bool(this->varVehiclesTransform()->size()) || this->stateTextureMesh()->isEmpty())
			return;

		auto&& config = this->varConfiguration()->getValue();

		auto rigidInfo = config.varRigidBodyConfigs();

		// **************************** Create RigidBody  **************************** //
		auto instances = this->varVehiclesTransform();
		int maxGroup = 0;

		for (auto rigid = rigidInfo->begin(); rigid != rigidInfo->end(); rigid++)
		{
			auto configGroup = rigidInfo->getElement(rigid).varConfigGroup()->getValue();
			if (configGroup > maxGroup)
				maxGroup = configGroup;
		}


		int j = 0;
		for (auto it = varVehiclesTransform()->begin(); it != varVehiclesTransform()->end(); it++, j++)
		{
			auto instance = instances->getElement(it);

			std::vector<std::shared_ptr<PdActor>> Actors;
			Actors.resize(rigidInfo->size());
			int i = -1;
			std::map<std::string, int> rigidName2Id;

			for (auto rigidIterator = rigidInfo->begin(); rigidIterator != rigidInfo->end(); rigidIterator++)
			{
				i++;
				auto rigid = rigidInfo->getElement(rigidIterator);
				
				rigidName2Id[rigid.varShapeName()->getValue()] = i;

				int visualId = -1;
				std::shared_ptr<Shape> visualShapePtr = NULL;
				if (rigid.varVisualShapeIds()->size())
				{
					int validIndex = int(texMesh->shapes().size()) - 1;

					if (rigid.varVisualShapeIds()->size())
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
				rigidbody.angle = rigid.varAngel()->getValue() * Quat<Real>(instance.rotation());
				rigidbody.linearVelocity = rigid.varLinearVelocity()->getValue();
				rigidbody.angularVelocity = rigid.varAngularVelocity()->getValue();

				if (!visualShapePtr)
				{
					rigidbody.position = rigid.varPosition()->getValue();
				}
				else
				{
					rigidbody.position = visualShapePtr->boundingTransform.translation() + rigidbody.position;
				}

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

					//ma
					bool matReady = false;
					if (!element.varMaAssetName()->getValue().empty())
					{
						auto maName = element.varMaAssetName()->getValue();
						const float scaleFactor = rigid.varScale()->getValue().x;

						auto vIter = Vertices.find(maName);
						auto eIter = Edges.find(maName);
						auto fIter = Faces.find(maName);

						matReady = vIter != Vertices.end() && eIter != Edges.end() && fIter != Faces.end();

						if (matReady)
						{
							const float scale2 = scaleFactor * scaleFactor;
							const float scale3 = scale2 * scaleFactor;
							const float scale5 = scale3 * scale2;
							MedialConeInfo medalcone;
							MedialSlabInfo medalslab;

							rigidbody.mass = mVolume[maName] * scale3 * element.varDensity()->getValue();
							rigidbody.inertia = this->mInertialMatrix[maName] * (scale5 * element.varDensity()->getValue());
							rigidbody.mass = mVolume[maName] * scale3 * element.varDensity()->getValue();
							rigidbody.friction = rigid.varFriction()->getValue() == -1 ? this->varFrictionCoefficient()->getValue() : rigid.varFriction()->getValue();
							rigidbody.position = rigid.varPosition()->getValue();
							rigidbody.angle = rigid.varAngel()->getValue();
							rigidbody.linearVelocity = rigid.varLinearVelocity()->getValue();
							rigidbody.angularVelocity = rigid.varAngularVelocity()->getValue();
							rigidbody.motionType = ToBodyType(rigid.varMotionType()->currentKey());
							rigidbody.bodyId = j * maxGroup + rigid.varConfigGroup()->getValue();;

							Actors[i] = this->createRigidBody(rigidbody, false);

							if (Actors[i] != nullptr)
							{
								auto& vertices = this->Vertices[maName];
								auto& edges = this->Edges[maName];
								auto& faces = this->Faces[maName];
								for (size_t j = 0; j < edges.size(); j++)
								{
									Vec2i edge = edges[j];
									if (edge[0] >= vertices.size() || edge[1] >= vertices.size())
									{
										std::cerr << "ERROR load edge" << std::endl;
										continue;
									}
									medalcone.v[0] = (Vec3f(vertices[edge[0]][0], vertices[edge[0]][1], vertices[edge[0]][2]) - mBaryCenter[maName]) * scaleFactor;
									medalcone.v[1] = (Vec3f(vertices[edge[1]][0], vertices[edge[1]][1], vertices[edge[1]][2]) - mBaryCenter[maName]) * scaleFactor;
									medalcone.radius[0] = vertices[edge[0]][3] * scaleFactor;
									medalcone.radius[1] = vertices[edge[1]][3] * scaleFactor;
									this->bindMedialCone(Actors[i], medalcone);
								}

								for (size_t j = 0; j < faces.size(); j++)
								{
									Vec3i face = faces[j];
									if (face[0] >= vertices.size() || face[1] >= vertices.size() || face[2] >= vertices.size())
									{
										std::cerr << "ERROR load face" << std::endl;
										continue;
									}


									medalslab.v[0] = (Vec3f(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2]) - mBaryCenter[maName]) * scaleFactor;
									medalslab.v[1] = (Vec3f(vertices[face[1]][0], vertices[face[1]][1], vertices[face[1]][2]) - mBaryCenter[maName]) * scaleFactor;
									medalslab.v[2] = (Vec3f(vertices[face[2]][0], vertices[face[2]][1], vertices[face[2]][2]) - mBaryCenter[maName]) * scaleFactor;
									medalslab.radius[0] = vertices[face[0]][3] * scaleFactor;
									medalslab.radius[1] = vertices[face[1]][3] * scaleFactor;
									medalslab.radius[2] = vertices[face[2]][3] * scaleFactor;
									this->bindMedialSlab(Actors[i], medalslab);
								}
							}

							visualId = mName2texMeshID[maName][0];
						}
						else
							continue;
					}
					else		// BasicShape 
					{
						rigidbody.position = Quat<Real>(instance.rotation()).rotate(rigidbody.position) + instance.translation();

						rigidbody.offset = rigid.varOffset()->getValue();
						rigidbody.inertia = rigid.varInertia()->getValue();
						rigidbody.friction = rigid.varFriction()->getValue() == -1 ? this->varFrictionCoefficient()->getValue() : rigid.varFriction()->getValue();
						rigidbody.restitution = rigid.varRestitution()->getValue();
						rigidbody.motionType = ToBodyType(rigid.varMotionType()->currentKey());
						rigidbody.shapeType = ElementType::ET_COMPOUND;
						rigidbody.collisionMask = ToCollisionMask(rigid.varCollisionMask()->currentKey());

						Actors[i] = this->createRigidBody(rigidbody);

						//Basic Shape
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

							if (Length == 0)
							{
								std::vector<Vector<Real, 3>> v[4];
								currentTet.v[0] = (visualShapePtr ? down : Vec3f(0));
								currentTet.v[1] = (visualShapePtr ? Vec3f(up.x, down.y, down.z) : Vec3f(1, 0, 0));
								currentTet.v[2] = (visualShapePtr ? Vec3f(up.x, down.y, up.z) : Vec3f(0, 1, 0));
								currentTet.v[3] = (visualShapePtr ? Vec3f(up) : Vec3f(0, 0, 1));
							}
							else if (Length > 0)
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
								else if (element.varCapsuleLength()->getValue() != 0)
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

				}

				if (visualId != -1 && Actors[i] != NULL)
				{
					////bindShapetoActor
					this->bindShape(Actors[i], Pair<uint, uint>(visualId, j), rigid.varScale()->getValue());
				}
			}

			auto jointInfo = config.varJointConfigs();
			for (auto jointIterator = jointInfo->begin(); jointIterator != jointInfo->end(); jointIterator++)
			{
				auto jointDetail = jointInfo->getElement(jointIterator);
				//Actor
				auto type = jointDetail.varJointType()->currentKey();
				auto A = jointDetail.varAShapeName()->getValue();
				auto B = jointDetail.varBShapeName()->getValue();
				auto AItor= rigidName2Id.find(A);
				auto BItor= rigidName2Id.find(B);
				if (AItor == rigidName2Id.end() || BItor == rigidName2Id.end())
					continue;

				int first = AItor->second;
				int second = BItor->second;
				Real speed = jointDetail.varMoter()->getValue();
				auto axis = Quat1f(instance.rotation()).rotate(jointDetail.varAxis()->getValue());
				auto anchorOffset = jointDetail.varAnchorPoint()->getValue();
				auto relative = jointDetail.varRelativeAnchorPoint()->getValue();

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


				if (type == JOINT_Hinge)
				{
					auto& hingeJoint = this->createHingeJoint(Actors[first], Actors[second]);
					hingeJoint.setAnchorPoint(relative ? (Actors[first]->center + anchorOffset) : anchorOffset);
					hingeJoint.setAxis(axis);
					if (jointDetail.varUseMoter()->getValue())
						hingeJoint.setMoter(speed);
					if (jointDetail.varUseRange()->getValue())
						hingeJoint.setRange(jointDetail.varRange()->getValue().x, jointDetail.varRange()->getValue().y);

				}
				if (type == JOINT_Slider)
				{
					auto& sliderJoint = this->createSliderJoint(Actors[first], Actors[second]);
					sliderJoint.setAnchorPoint((Actors[first]->center + Actors[first]->center) / 2 + anchorOffset);
					sliderJoint.setAxis(axis);
					if (jointDetail.varUseMoter()->getValue())
						sliderJoint.setMoter(speed);
					if (jointDetail.varUseRange()->getValue())
						sliderJoint.setRange(jointDetail.varRange()->getValue().x, jointDetail.varRange()->getValue().y);
				}
				if (type == JOINT_Fixed)
				{
					auto& fixedJoint1 = this->createFixedJoint(Actors[first], Actors[second]);
					fixedJoint1.setAnchorPoint(relative ? ((Actors[first]->center + Actors[first]->center) / 2 + anchorOffset) : anchorOffset + instance.translation());
				}
				if (type == JOINT_Point)
				{
					auto& pointJoint = this->createPointJoint(Actors[first]);
					pointJoint.setAnchorPoint(relative ? (Actors[first]->center + anchorOffset) : anchorOffset + instance.translation());
				}
				if (type == JOINT_BallAndSocket)
				{
					auto& ballAndSocketJoint = this->createBallAndSocketJoint(Actors[first], Actors[second]);
					ballAndSocketJoint.setAnchorPoint(relative ? ((Actors[first]->center + Actors[first]->center) / 2 + anchorOffset) : anchorOffset + instance.translation());
				}
			}
		}
		
	}
	template<typename TDataType>
	void ConfigurableBody<TDataType>::onTexMeshLoad()
	{
		auto&& config = this->varConfiguration()->getValue();

		auto rigidInfo = config.varRigidBodyConfigs();
		auto assetInfo = config.varAssetConfigs();
		auto jointInfo = config.varJointConfigs();

		std::vector<FilePath> texMeshFiles;
		std::vector<std::string> assetNames;
		std::vector<FilePath> maFiles;

		for (auto it = assetInfo->begin(); it != assetInfo->end(); it++)
		{
			auto asset = assetInfo->getElement(it);
			if (asset.isValid()) 
			{
				texMeshFiles.push_back(asset.varTexMeshPath()->getValue());
				maFiles.push_back(asset.varMaPath()->getValue());
				assetNames.push_back(asset.varAssetName()->getValue());
			}
		}

		if (texMeshFiles.size())
		{
			loadTextureMeshFromFiles(
				this->stateTextureMesh()->getDataPtr(),
				texMeshFiles,
				assetNames,
				mBaryCenter,
				mVolume,
				mInertialMatrix,
				mName2texMeshID,
				true
			);
		}
		if (maFiles.size()) 
		{
			for (size_t i = 0; i < maFiles.size(); i++)
			{
				loadMa(assetNames[i], maFiles[i].string());
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


	template<typename TDataType>
	bool ConfigurableBody<TDataType>::loadMa(std::string name, std::string file_path)
	{
		std::vector<Vec4f> vertices;
		std::vector<Vec2i> edges;
		std::vector<Vec3i>	faces;
		std::string filename = file_path;
		
		bool is_absolute = false;
		if (!file_path.empty())
		{
			if (file_path.size() >= 3 && std::isalpha(file_path[0]) && file_path[1] == ':')
			{
				char sep = file_path[2];
				if (sep == '\\' || sep == '/')
				{
					is_absolute = true;
				}
			}
			else if (file_path.size() >= 2 && file_path[0] == '\\' && file_path[1] == '\\')
			{
				is_absolute = true;
			}
		}

		if (is_absolute)
		{
			filename = file_path;
		}
		else
		{
			std::string asset_root = getAssetPath();
			if (!asset_root.empty())
			{
				char last = asset_root.back();
				if (last != '\\' && last != '/')
				{
					asset_root += "\\";
				}
			}
			filename = asset_root + file_path;
		}
		
		std::ifstream inputFile(filename);
		std::string line;
		int num_vertices = 0, num_edges = 0, num_faces = 0;

		if (!inputFile.is_open())
		{
			std::cerr << "ERROR MA FILE: cannot open " << filename << std::endl;
			return false;
		}

		if (std::getline(inputFile, line))
		{
			std::istringstream iss(line);
			if (!(iss >> num_vertices >> num_edges >> num_faces)) {
				std::cerr << "ERROR MA FILE: invalid header in " << filename << std::endl;
				inputFile.close();
				return false;
			}
			std::cout << "num Of vertices : " << num_vertices
				<< ", num Of edges : " << num_edges
				<< ", num Of faces : " << num_faces << std::endl;
		}
		else
		{
			std::cerr << "ERROR MA FILE: empty file " << filename << std::endl;
			inputFile.close();
			return false;
		}

		int current_vertex_count = 0;
		int current_edge_count = 0;
		int current_face_count = 0;

		while (std::getline(inputFile, line))
		{
			if (line.empty() || line[0] == '#')
			{
				continue;
			}

			std::istringstream iss(line);
			char type;
			iss >> type;

			if (type == 'v' && current_vertex_count < num_vertices)
			{
				Vec4f vertex;
				if (iss >> vertex[0] >> vertex[1] >> vertex[2] >> vertex[3])
				{
					vertex = vertex;
					vertices.push_back(vertex);
					current_vertex_count++;
				}
				else {
					std::cerr << "ERROR load vertex" << std::endl;
				}
			}
			else if (type == 'e' && current_edge_count < num_edges)
			{
				Vec2i edge;
				if (iss >> edge[0] >> edge[1])
				{
					if (edge[0] >= 0 && edge[0] < num_vertices && edge[1] >= 0 && edge[1] < num_vertices)
					{
						edges.push_back(edge);
						current_edge_count++;
					}
					else {
						std::cerr << "ERROR load edge" << std::endl;
					}
				}
				else {
					std::cerr << "ERROR load edge" << std::endl;
				}
			}
			else if (type == 'f' && current_face_count < num_faces)
			{
				Vec3i face;
				if (iss >> face[0] >> face[1] >> face[2])
				{
					if (face[0] >= 0 && face[0] < num_vertices && face[1] >= 0 && face[1] < num_vertices && face[2] >= 0 && face[2] < num_vertices)
					{
						faces.push_back(face);
						current_face_count++;
					}
					else {
						std::cerr << "ERROR load face" << std::endl;
					}
				}
				else {
					std::cerr << "ERROR load face" << std::endl;
				}
			}
		}
		inputFile.close();

		this->Vertices[name] = std::move(vertices);
		this->Edges[name] = std::move(edges);
		this->Faces[name] = std::move(faces);

		return num_vertices == current_vertex_count
			&& num_edges == current_edge_count
			&& num_faces == current_face_count;
	}

	DEFINE_CLASS(ConfigurableBody);



}
