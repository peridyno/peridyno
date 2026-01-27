#include "MujocoXMLLoader.h"
#include "Mapping/TextureMeshToTriangleSet.h"
#include "helpers/tinyobj_helper.h"
#include "GLSurfaceVisualModule.h"
#include "GLPhotorealisticRender.h"
#include "Topology/HierarchicalModel.h"

#include <Mapping/DiscreteElementsToTriangleSet.h>
#include <Mapping/ContactsToEdgeSet.h>
#include <Mapping/ContactsToPointSet.h>

#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"
#include "GLPointVisualModule.h"
#include "helpers/stlreader_helper.h"

namespace dyno
{

	template<typename TDataType>
	MujocoXMLLoader<TDataType>::MujocoXMLLoader() 
	{
		this->varNum()->setRange(0,30);


		auto convert2TriSet = std::make_shared<TextureMeshToTriangleSet<TDataType>>();
		this->stateTextureMesh()->connect(convert2TriSet->inTextureMesh());


		auto callback = std::make_shared<FCallBackFunc>(std::bind(&MujocoXMLLoader<TDataType>::filePathChanged, this));
		this->varFilePath()->attach(callback);
		this->varNum()->attach(callback);

		auto callbackTransform = std::make_shared<FCallBackFunc>(std::bind(&MujocoXMLLoader<TDataType>::updateTransform, this));
		this->varLocation()->attach(callbackTransform);
		this->varScale()->attach(callbackTransform);
		this->varRotation()->attach(callbackTransform);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		auto tsRender = std::make_shared<GLSurfaceVisualModule>();
		tsRender->setVisible(true);
		this->stateTriangleSet()->connect(tsRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(tsRender);

		this->stateTriangleSet()->promoteOuput();


		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		this->stateTopology()->connect(mapper->inDiscreteElements());
		this->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->setColor(Color(1, 1, 0));
		sRender->setAlpha(0.5f);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);

		//Visualize contact points
		auto elementQuery = std::make_shared<NeighborElementQuery<DataType3f>>();
		this->stateTopology()->connect(elementQuery->inDiscreteElements());
		this->stateCollisionMask()->connect(elementQuery->inCollisionMask());
		this->graphicsPipeline()->pushModule(elementQuery);

		auto contactPointMapper = std::make_shared<ContactsToPointSet<DataType3f>>();
		elementQuery->outContacts()->connect(contactPointMapper->inContacts());
		this->graphicsPipeline()->pushModule(contactPointMapper);

		auto pointRender = std::make_shared<GLPointVisualModule>();
		pointRender->setColor(Color(1, 0, 0));
		pointRender->varPointSize()->setValue(0.01f);
		contactPointMapper->outPointSet()->connect(pointRender->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender);

	}

	template<typename TDataType>
	MujocoXMLLoader<TDataType>::~MujocoXMLLoader()
	{

	}




	template<typename TDataType>
	bool MujocoXMLLoader<TDataType>::filePathChanged()
	{

		mXMLAssetMeshes.clear();
		mXMLDefaultClass.clear();
		mXMLBody.clear();
		this->stateTextureMesh()->getDataPtr()->clear();


		tinyxml2::XMLDocument doc;

		auto filepath = this->varFilePath()->getValue();
		auto file = filepath.string();
		FilePath a;
		auto path = filepath.path().string();
		auto parentPath = filepath.path().parent_path().string();
		printf("\n\n***************************************\n");

		auto result = doc.LoadFile(file.c_str());

		auto triSet = this->stateTriangleSet()->getDataPtr();
		if (result != tinyxml2::XML_SUCCESS)
		{
			return false;
		}
	
		//tinyxml2::XMLElement* root = doc.RootElement();

		// <mujoco> 
		tinyxml2::XMLElement* mujocoElement = doc.FirstChildElement("mujoco");
		if (!mujocoElement)
		{
			std::cerr << "No <mujoco> element found\n";

		}
		
		// <compiler>
		tinyxml2::XMLElement* compilerElement = mujocoElement->FirstChildElement("compiler");
		if (!compilerElement)
		{
			std::cerr << "No <compiler> element found\n";
		}
		else
		{
			parseCompilerElement(compilerElement);
		}


		std::string assetPath = parentPath + "\\" + this->complier.meshdir + "\\";
		std::cout << "parentPath : " << parentPath << "\n";
		std::cout << "assetPath : " << assetPath << "\n";

		// <default> 
		tinyxml2::XMLElement* defaultElement = mujocoElement->FirstChildElement("default");
		if (!defaultElement)
		{
			std::cerr << "No <default> element found\n";
		}
		else 
		{	
			parseDefaultElement(defaultElement);
		}

		// <asset> 
		tinyxml2::XMLElement* assetElement = mujocoElement->FirstChildElement("asset");
		printf("=======  Asset  =========  \n");

		parseAsset(assetElement, assetPath);
		
		// <worldbody> 
		tinyxml2::XMLElement* worldBodyElement = mujocoElement->FirstChildElement("worldbody");
		if (worldBodyElement)
		{
			parseWorldBodyElement(worldBodyElement);
		}
		else 
		{
			std::cerr << "No <worldBodyElement> element found\n";

		}

		std::vector<Vec3f> points;
		std::vector<uint> shapeIDs;
		int triIDOffset = 0;

		printf("\n\n =========== Build Model ============\n\n");
		int bodyNum = 0;
		std::shared_ptr<TextureMesh> texMesh = this->stateTextureMesh()->getDataPtr();
		for (auto body : mXMLBody)
		{	
			bodyNum++;
			if (true)
			{
				std::cout << "************** Body: " << body->name.value() << "\n";
				std::cout << "	Combine: " << body->name.value() << "\n";
				if (body->bodyGeoms.size())
				{
					Mesh combine;
					bool hasShape = false;
					for (int i = 0; i < body->bodyGeoms.size(); i++)
					{
						if (body->bodyGeoms[i]->meshName.has_value())
						{
							std::cout << "		" << body->bodyGeoms[i]->meshName.value() << "\n";
							Mesh meshB = getMeshByName(body->bodyGeoms[i]->meshName.value());
							if (body->bodyGeoms[i]->quat.has_value())
								rotateVertices(meshB.vertices, body->bodyGeoms[i]->quat.value());
							if(body->realPos.has_value())
								offsetVertices(meshB.vertices,body->realPos.value());
							if(body->bodyGeoms[i]->pos.has_value())
								offsetVertices(meshB.vertices, body->bodyGeoms[i]->pos.value());

							combine = mergeMesh(combine, meshB);
							hasShape = true;
						}
					}

					if (hasShape)
					{
						int shapeId = texMesh->shapes().size();
						std::vector<uint> tempShapeIds;
						tempShapeIds.resize(combine.vertices.size(), shapeId);
						points.insert(points.end(), combine.vertices.begin(), combine.vertices.end());
						shapeIDs.insert(shapeIDs.end(), tempShapeIds.begin(), tempShapeIds.end());

						offsetTriangleIndex(combine.triangles, triIDOffset);
						auto shape = std::make_shared<Shape>();
						shape->vertexIndex.assign(combine.triangles);
						texMesh->shapes().push_back(shape);

						body->shapeId = shapeId;
					}
					triIDOffset = points.size();
				}
			}

		}
		texMesh->geometry()->vertices().assign(points);
		texMesh->geometry()->shapeIds().assign(shapeIDs);

		texMesh->updateTexMeshBoundingBox();

		//ToCenter
		if (true)
		{
			initialShapeCenter = texMesh->updateTexMeshBoundingBox();

			DArray<Vec3f> unCenterPosition;
			DArray<Vec3f> d_ShapeCenter;

			d_ShapeCenter.assign(initialShapeCenter);	// Used to "ToCenter"
			unCenterPosition.assign(this->stateTextureMesh()->getDataPtr()->geometry()->vertices());
			CArray<Vec3f> cshapeCenter;
			cshapeCenter.assign(d_ShapeCenter);

			HierarchicalScene helper;
			helper.shapeToCenter(
				unCenterPosition,
				texMesh->geometry()->vertices(),
				texMesh->geometry()->shapeIds(),
				d_ShapeCenter
			);


		}

		return true;
	}

	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::updateTransform() 
	{
	
	}

	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::parseDefaultElement(tinyxml2::XMLElement* element, int depth = 0, std::shared_ptr<XMLClass> parent = nullptr)
	{
		if (element == nullptr)
			return;

		auto xmlClass = std::make_shared<XMLClass>();
		xmlClass->parent = parent;

		const char* classAttr = element->Attribute("class");
		if (classAttr) {
			std::cout << std::string(depth * 2, ' ') << "class: " << classAttr << std::endl;
			xmlClass->name = classAttr;
		};

		std::shared_ptr<XMLGeom> xmlGeom = std::make_shared<XMLGeom>();

		tinyxml2::XMLElement* geomElement = element->FirstChildElement("geom");
		if (geomElement != nullptr)
		{
			parseGeomElemet(geomElement, xmlGeom);
			coutGeomElemet(xmlGeom);
		}
		xmlClass->geomData = xmlGeom;

		std::shared_ptr<XMLJoint> xmlJoint = std::make_shared<XMLJoint>();

		tinyxml2::XMLElement* jointElement = element->FirstChildElement("joint");
		if (jointElement != nullptr)
		{
			parseJointElemet(jointElement, xmlJoint);
		}
		xmlClass->jointData = xmlJoint;

		std::shared_ptr<XMLSite> xmlSite = std::make_shared<XMLSite>();
		tinyxml2::XMLElement* siteElement = element->FirstChildElement("site");
		if (siteElement != nullptr)
		{
			parseSiteElemet(siteElement, xmlSite);
		}

		tinyxml2::XMLElement* childElement = element->FirstChildElement("default");
		while (childElement) {
			parseDefaultElement(childElement, depth + 1, xmlClass);
			childElement = childElement->NextSiblingElement("default");

		}

		mXMLDefaultClass.push_back(xmlClass);
	}
	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::parseCompilerElement(tinyxml2::XMLElement* element)
	{
		const char* angleAttr = element->Attribute("angle");
		if (angleAttr)
		{
			this->complier.angle = angleAttr;
		}

		const char* meshDirAttr = element->Attribute("meshdir");
		if (meshDirAttr)
		{
			this->complier.meshdir = meshDirAttr;
		}

		const char* autolimitsAttr = element->Attribute("autolimits");
		if (autolimitsAttr)
		{
			if(autolimitsAttr == "true")
				this->complier.autolimits = true;
			else
				this->complier.autolimits = false;

		}
	}
	


	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::parseWorldBodyElement(tinyxml2::XMLElement* element)
	{
		if (element == nullptr)
			return;

		tinyxml2::XMLElement* bodyElement = element->FirstChildElement("body");
		if (bodyElement != nullptr)
		{
			parseBodyElement(bodyElement);

		}

	}

	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::parseAsset(tinyxml2::XMLElement* assetElement,const std::string& assetPath)
	{
		auto texMesh = this->stateTextureMesh()->getDataPtr();
		if (!assetElement)
		{
			std::cerr << "No <asset> element found\n";

		}
		else
		{
			//  <mesh> 
			for (tinyxml2::XMLElement* meshElement = assetElement->FirstChildElement("mesh");
				meshElement != nullptr;
				meshElement = meshElement->NextSiblingElement("mesh"))
			{
				const char* name = meshElement->Attribute("name");
				const char* file = meshElement->Attribute("file");
				XMLMesh mesh;
				if (file)
				{
					std::string filename = std::string(file);
					mesh.file = filename;

					auto ext = getFileExtension(mesh.file.value());
					if (ext == std::string(".obj"))
					{
						std::string currentFile = assetPath + mesh.file.value();
						auto load = loadObj(mesh.vertices, mesh.triangles, currentFile);
						convertVertices(mesh.vertices);
					}
					if (ext == std::string(".STL")|| ext == std::string(".stl"))
					{
						std::string currentFile = assetPath + mesh.file.value();
						auto load = loadStl(mesh.vertices, mesh.triangles, currentFile);
						convertVertices(mesh.vertices);
					}

					if (name)
						mesh.name = name;
					else
					{
						size_t pos = filename.find('.');
						mesh.name = (pos == std::string::npos) ? filename : filename.substr(0, pos);
					}
				}
				mXMLAssetMeshes.push_back(mesh);
			}

			for (size_t i = 0; i < mXMLAssetMeshes.size(); i++)
			{
				if (mXMLAssetMeshes[i].name.has_value())
					std::cout << mXMLAssetMeshes[i].name.value() << "\n";
			}

		}


	}

	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::parseBodyElement(tinyxml2::XMLElement* element, int depth = 0, std::shared_ptr<XMLBody> parent = nullptr) 
	{
		if (element == nullptr)
			return;
		printf(" ============  Body ============\n");

		auto xmlBody = std::make_shared<XMLBody>();
		xmlBody->parentBody = parent;

		const char* nameAttr = element->Attribute("name");
		if (nameAttr) {
			std::cout << std::string(depth * 2, ' ') << "name: " << nameAttr << std::endl;
			xmlBody->name = nameAttr;
		};

		const char* posAttr = element->Attribute("pos");
		if (posAttr) {
			xmlBody->pos = convertCoord(ParseVec3f(posAttr));
		}
		else
		{
			xmlBody->pos = Vec3f(0);
		}

		if (xmlBody->parentBody != nullptr)
			xmlBody->realPos = xmlBody->parentBody->realPos.value() + xmlBody->pos.value();
		else
			xmlBody->realPos = xmlBody->pos;


		const char* childClassAttr = element->Attribute("childclass");
		if (childClassAttr) {

			for (auto it : mXMLDefaultClass)
			{
				if (it->name == std::string(childClassAttr)) 
				{
					xmlBody->childClass = it;
					break;
				}
			}
			

		};

		tinyxml2::XMLElement* inertialElement = element->FirstChildElement("inertial");
		if (inertialElement != nullptr)
		{
			const char* posAttr = inertialElement->Attribute("pos");
			if (posAttr) {
				xmlBody->inertialPos = convertCoord(ParseVec3f(posAttr));
			};

			const char* quatAttr = inertialElement->Attribute("quat");
			if (quatAttr) {
				auto q = decodeVec4f(quatAttr);
				xmlBody->quat = convertQuat(Quat<Real>(q.x,q.y,q.z,q.w));
			};
			
			const char* massAttr = inertialElement->Attribute("mass");
			if (massAttr) {
				xmlBody->mass = std::stod(massAttr);
			};
		}

		printf("     ============  Mesh ============\n");
		for (tinyxml2::XMLElement* geomElement = element->FirstChildElement("geom");
			geomElement != nullptr;
			geomElement = geomElement->NextSiblingElement("geom"))
		{
			
			std::shared_ptr<XMLGeom> geom = std::make_shared<XMLGeom>();
			parseGeomElemet(geomElement, geom);

			xmlBody->bodyGeoms.push_back(geom);
			

		}

		tinyxml2::XMLElement* jointElement = element->FirstChildElement("joint");
		if (jointElement != nullptr) 
		{
			auto bodyJoint = std::make_shared<XMLJoint>();
			parseJointElemet(jointElement, bodyJoint);

			xmlBody->joint = bodyJoint;
		}

		//const char* nameAttr = element->Attribute("name");
		//xmlBody->meshName.push_back();

		tinyxml2::XMLElement* childElement = element->FirstChildElement("body");
		while (childElement) {
			parseBodyElement(childElement, depth + 1, xmlBody);
			childElement = childElement->NextSiblingElement("body");

		}

		mXMLBody.push_back(xmlBody);
	}



	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::parseGeomElemet(tinyxml2::XMLElement* geomElement, std::shared_ptr<XMLGeom> geom)
	{
		if (geomElement == nullptr)
			return;

		const char* type = geomElement->Attribute("type");
		if (type)
		{
			if (strcmp(type, "plane") == 0) {
				geom->type = XMLGeomType::plane;
			}
			else if (strcmp(type, "hfield") == 0) {
				geom->type = XMLGeomType::hfield;
			}
			else if (strcmp(type, "sphere") == 0) {
				geom->type = XMLGeomType::sphere;
			}
			else if (strcmp(type, "capsule") == 0) {
				geom->type = XMLGeomType::capsule;
			}
			else if (strcmp(type, "ellipsoid") == 0) {
				geom->type = XMLGeomType::ellipsoid;
			}
			else if (strcmp(type, "cylinder") == 0) {
				geom->type = XMLGeomType::cylinder;
			}
			else if (strcmp(type, "box") == 0) {
				geom->type = XMLGeomType::box;
			}
			else if (strcmp(type, "mesh") == 0) {
				geom->type = XMLGeomType::mesh;
			}
			else if (strcmp(type, "sdf") == 0) {
				geom->type = XMLGeomType::sdf;
			}
		}

		const char* size = geomElement->Attribute("size");
		if (size)
		{
			int count = countFields(size);
			std::cout << "count ::" << count << "\n";
			switch (count) 
			{
			case 1:
				geom->size = Vec3f(std::stod(size),-1,-1);
				break;
			case 2: 
			{
				Vec2f temp = ParseVec2f(size);
				geom->size = Vec3f(temp[0], temp[1], -1);

				break;
			}
			case 3:
				geom->size = convertCoord(ParseVec3f(size));
				break;
			default:
				break;
			}
		}


		const char* contype = geomElement->Attribute("contype");
		if (contype)
			geom->contype = std::stoi(contype);

		const char* conaffinity = geomElement->Attribute("conaffinity");
		if (contype)
			geom->conaffinity = std::stoi(conaffinity);

		const char* group = geomElement->Attribute("group");
		if (group)
			geom->group = std::stoi(group);

		const char* density = geomElement->Attribute("density");
		if (density)
			geom->density = std::stod(density);

		const char* pos = geomElement->Attribute("pos");
		if (pos)
			geom->pos = convertCoord(ParseVec3f(pos));

		const char* quat = geomElement->Attribute("quat");
		if (quat) 
		{
			Vec4f temp = ParseVec4f(quat);
			geom->quat = convertQuat(Quat<Real>(temp[0],temp[1],temp[2],temp[3]));

		}

		const char* rgba = geomElement->Attribute("rgba");
		if (rgba)
		{
			geom->rgba = ParseVec4f(rgba);
		}

		const char* mass = geomElement->Attribute("mass");
		if (mass)
			geom->mass = std::stod(mass);

		const char* friction = geomElement->Attribute("friction");
		if (friction)
			geom->friction = convertCoord(ParseVec3f(friction));

		const char* mesh = geomElement->Attribute("mesh");
		if (mesh) 
		{
			geom->meshName = std::string(mesh);
			geom->type = XMLGeomType::mesh;
		}

		const char* classAttr = geomElement->Attribute("class");
		if (classAttr)
		{
			std::string className = std::string(classAttr);
			
			for (auto it : mXMLDefaultClass)
			{
				if (it->name == className) 
				{
					geom->XmlClass = it;
					parseGeomFromXML(*it, geom);

					break;
				}
			}
		}


	}


	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::parseJointElemet(tinyxml2::XMLElement* jointElement, std::shared_ptr<XMLJoint> joint)
	{
		if (jointElement == nullptr)
			return;

		const char* type = jointElement->Attribute("type");
		if (type)
		{
			if (strcmp(type, "ball") == 0) {
				joint->type = XMLJointType::ball;
			}
			else if (strcmp(type, "free") == 0) {
				joint->type = XMLJointType::free;
			}
			else if (strcmp(type, "hinge") == 0) {
				joint->type = XMLJointType::hinge;
			}
			else if (strcmp(type, "slide") == 0) {
				joint->type = XMLJointType::slide;
			}
		}


		const char* damping = jointElement->Attribute("damping");
		if (damping)
			joint->damping = std::stod(damping);

		const char* axis = jointElement->Attribute("axis");
		if (axis)
		{
			joint->axis = convertCoord(ParseVec3f(axis));
			std::cout << "axis :" << joint->axis.value()[0] << ", " << joint->axis.value()[1] << ", " << joint->axis.value()[2] << "\n";

		}


		const char* range = jointElement->Attribute("range");
		if (range)
		{
			joint->range = ParseVec2f(range);
			std::cout << "range :" << joint->range.value()[0] << ", " << joint->range.value()[1] << "\n";

		}

		const char* xmlclass = jointElement->Attribute("class");
		if (xmlclass)
		{
			for (auto it : this->mXMLDefaultClass)
			{
				if (std::string(xmlclass) == it->name) 
				{
					joint->XmlClass = it;
					parseJointFromXML(*it, joint);

					break;
				}
			}

		}

	}


	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::parseSiteElemet(tinyxml2::XMLElement* geomElement, std::shared_ptr<XMLSite> site)
	{
		if (geomElement == nullptr)
			return;

		const char* type = geomElement->Attribute("type");
		if (type)
		{
			if (strcmp(type, "plane") == 0) {
				site->type = XMLGeomType::plane;
			}
			else if (strcmp(type, "hfield") == 0) {
				site->type = XMLGeomType::hfield;
			}
			else if (strcmp(type, "sphere") == 0) {
				site->type = XMLGeomType::sphere;
			}
			else if (strcmp(type, "capsule") == 0) {
				site->type = XMLGeomType::capsule;
			}
			else if (strcmp(type, "ellipsoid") == 0) {
				site->type = XMLGeomType::ellipsoid;
			}
			else if (strcmp(type, "cylinder") == 0) {
				site->type = XMLGeomType::cylinder;
			}
			else if (strcmp(type, "box") == 0) {
				site->type = XMLGeomType::box;
			}
			else if (strcmp(type, "mesh") == 0) {
				site->type = XMLGeomType::mesh;
			}
			else if (strcmp(type, "sdf") == 0) {
				site->type = XMLGeomType::sdf;
			}
		}

		const char* group = geomElement->Attribute("group");
		if (group)
			site->group = std::stoi(group);

		const char* size = geomElement->Attribute("size");
		if (size) 
		{
			int count = countFields(size);
		}

		const char* rgba = geomElement->Attribute("rgba");
		if (rgba)
		{
			site->rgba = ParseVec4f(rgba);

			std::cout << "rgba :" << site->rgba.value()[0] << ", " << site->rgba.value()[1] << ", " << site->rgba.value()[2] << ", " << site->rgba.value()[3] << "\n";
		}
	}

	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::coutGeomElemet(std::shared_ptr<XMLGeom> geom)
	{
		if (geom->name.has_value())
			std::cout << " name : " << geom->name.value() << " ";

		if (geom->type.has_value())
		{
			switch (geom->type.value())
			{
			case plane:
				std::cout << "plane ";
				break;
			case hfield:
				std::cout << "hfield ";
				break;
			case sphere:
				std::cout << "sphere ";
				break;
			case capsule:
				std::cout << "capsule ";
				break;
			case ellipsoid:
				std::cout << "ellipsoid ";
				break;
			case cylinder:
				std::cout << "cylinder ";
				break;
			case box:
				std::cout << "box ";
				break;
			case mesh:
				std::cout << "mesh ";
				break;
			case sdf:
				std::cout << "sdf ";
				break;
			default:
				break;
			}
		}
		if (geom->size.has_value())
			std::cout << " size : " << geom->size.value() << " ";
		if (geom->pos.has_value())
			printf("pos: %f,%f,%f ", geom->pos.value()[0], geom->pos.value()[1], geom->pos.value()[2]);

		if (geom->group.has_value())
			std::cout << " group : " << geom->group.value() << " ";
		if (geom->contype.has_value())
			std::cout << " contype : " << geom->contype.value() << " ";
		if (geom->conaffinity.has_value())
			std::cout << " conaffinity : " << geom->conaffinity.value() << " ";
		if (geom->mass.has_value())
			std::cout << " mass : " << geom->mass.value() << " ";

		if (geom->friction.has_value())
			printf("friction: %f,%f,%f ", geom->friction.value()[0], geom->friction.value()[1], geom->friction.value()[2]);

		printf("\n");
	}

	template<typename TDataType>
	Mesh MujocoXMLLoader<TDataType>::mergeMesh(const Mesh& a,const Mesh& b)
	{
		int offset = a.vertices.size();

		Mesh c;
		Mesh b2;
		c.vertices.reserve(a.vertices.size() + b.vertices.size());
		c.vertices.insert(c.vertices.end(), a.vertices.begin(), a.vertices.end());
		c.vertices.insert(c.vertices.end(), b.vertices.begin(), b.vertices.end());
		for (auto it : b.triangles)
		{
			b2.triangles.push_back(TopologyModule::Triangle(it[0] + offset, it[1] + offset, it[2] + offset));
		}
		c.triangles.reserve(a.triangles.size() + b.triangles.size());
		c.triangles.insert(c.triangles.end(), a.triangles.begin(), a.triangles.end());
		c.triangles.insert(c.triangles.end(), b2.triangles.begin(), b2.triangles.end());

		return c;
	}

	template<typename TDataType>
	Mesh MujocoXMLLoader<TDataType>::getMeshByName(std::string name)
	{
		Mesh result;
		for (auto it : mXMLAssetMeshes)
		{
			if (it.name == name)
			{
				result = Mesh(it.vertices, it.triangles);
				break;
			}
		}
		return result;
	}

	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::offsetTriangleIndex(std::vector<TopologyModule::Triangle>& triangles, const int& offset)
	{
		for (size_t i = 0; i < triangles.size(); i++)
		{
			triangles[i] = TopologyModule::Triangle(triangles[i][0] + offset, triangles[i][1] + offset, triangles[i][2] + offset);

		}
	}
	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::offsetVertices(std::vector<Vec3f>& vertices, const Vec3f& offset)
	{
		for (size_t i = 0; i < vertices.size(); i++)
		{
			vertices[i] = vertices[i] + offset;
		}
	}
	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::rotateVertices(std::vector<Vec3f>& vertices, Quat<Real> q)
	{
		for (size_t i = 0; i < vertices.size(); i++)
		{
			vertices[i] = q.rotate(vertices[i]);
		}
	}

	template<typename TDataType>
	bool MujocoXMLLoader<TDataType>::isCollision(const XMLGeom& geom)
	{
		
		bool collision = true;
		if (geom.contype.has_value())
			if (geom.contype == 0)
				collision = false;

		return collision;
		

	}
	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::createRigidBodySystem()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();

		std::shared_ptr<TextureMesh> texMesh = this->stateTextureMesh()->getDataPtr();


		auto instances = this->varVehiclesTransform()->getValue();
		uint vehicleNum = instances.size();
		for (size_t i = 0; i < vehicleNum; i++)
		{
			for (auto body : mXMLBody)
				body->actor = nullptr;

			RigidBodyInfo rigidbody;
			rigidbody.bodyId = i;
			rigidbody.friction = this->varFrictionCoefficient()->getValue();
			rigidbody.angle = Quat1f(instances[i].rotation());

			for (auto body : mXMLBody)
			{


				Vec3f mujocoBodyPos = Vec3f(0);
				if (body->realPos.has_value())
					mujocoBodyPos = body->realPos.value();

				Vec3f offset = Vec3f(0);

				if (body->inertialPos.has_value())
					offset = body->inertialPos.value();

				rigidbody.offset = offset;

				body->massCenter = mujocoBodyPos + offset;

				for (auto geom : body->bodyGeoms)
				{
					if (isCollision(*geom))
					{

						if (body->realPos.has_value())
						{
							if (body->shapeId != -1)
							{
								rigidbody.position = Quat1f(instances[i].rotation()).rotate(texMesh->shapes()[body->shapeId]->boundingTransform.translation()) + instances[i].translation();
							}
							else
								rigidbody.position = body->massCenter.value();
						}

						Vec3f geomOffset = Vec3f(0);
						if (geom->pos.has_value())
						{
							if (body->shapeId != -1)
							{
								geomOffset = geom->pos.value() + (body->realPos.value() - texMesh->shapes()[body->shapeId]->boundingTransform.translation());
							}
							else
							{
								geomOffset = geom->pos.value();

							}
						}
						else
						{
							geomOffset = (body->realPos.value() - texMesh->shapes()[body->shapeId]->boundingTransform.translation());
						}

						Quat<Real> bodyRot;
						if (body->quat.has_value())
							bodyRot = body->quat.value();
						XMLGeomType a = XMLGeomType::box;
						switch (geom->type.value())//
						{
						case dyno::plane:
							break;
						case dyno::hfield:
							break;
						case dyno::sphere:
						{
							if (body->actor == nullptr)
								body->actor = this->createRigidBody(rigidbody);
							SphereInfo sphere;
							sphere.center = geomOffset;
							if (geom->size.has_value())
								sphere.radius = geom->size.value()[0];
							else
							{
								sphere.radius = 0.1;

							}
							this->bindSphere(body->actor, sphere);
							break;
						}
						case dyno::capsule:
							break;
						case dyno::ellipsoid:
							break;
						case dyno::cylinder:
						{
							if (body->actor == nullptr)
								body->actor = this->createRigidBody(rigidbody);

							CapsuleInfo capsule;
							capsule.radius = geom->size.value()[0];
							capsule.halfLength = geom->size.value()[1];

							capsule.center = geomOffset;

							if (geom->quat.has_value())
							{
								capsule.rot = geom->quat.value();
								std::cout << body->name.value();
							}
							this->bindCapsule(body->actor, capsule);
							break;
						}
						case dyno::box:
						{
							if (body->actor == nullptr)
								body->actor = this->createRigidBody(rigidbody);

							BoxInfo box;

							box.center = geomOffset;
							Vec3f tempSize = geom->size.value();
							box.halfLength = Vec3f(abs(tempSize.x), abs(tempSize.y), abs(tempSize.z));//* 0.7
							if (geom->quat.has_value())
							{
								box.rot = geom->quat.value();

							}
							this->bindBox(body->actor, box);
							break;
						}
						case dyno::mesh:
						{
							if (body->actor == nullptr)
								body->actor = this->createRigidBody(rigidbody);
							SphereInfo sphere;
							sphere.center = geomOffset;
							if (geom->size.has_value())
								sphere.radius = geom->size.value()[0];
							else
							{
								sphere.radius = 0.05;

							}
							this->bindSphere(body->actor, sphere);
							break;
						}
						case dyno::sdf:
							break;
						default:
							break;
						}

					}

				}
			}


			for (auto body : mXMLBody)
			{
				if (body->parentBody)
				{
					if (body->parentBody->actor && body->actor)
					{
						if (body->joint)
						{
							if (body->joint->type == XMLJointType::hinge)
							{
								auto& joint = this->createHingeJoint(body->parentBody->actor, body->actor);
								joint.setAnchorPoint(Quat1f(instances[i].rotation()).rotate(body->realPos.value()) + instances[i].translation());

								if (body->joint->axis.has_value())
									joint.setAxis(Quat1f(instances[i].rotation()).rotate(body->joint->axis.value()));
								else
									joint.setAxis(Quat1f(instances[i].rotation()).rotate(Vec3f(0, 0, 1)));

								if (body->joint->range.has_value())
									joint.setRange(body->joint->range.value()[0], body->joint->range.value()[1]);
							}
						}
					}
				}

				if (body->actor != nullptr && body->shapeId != -1)
				{
					this->bindShape(body->actor, Pair<uint, uint>(body->shapeId, i));
				}

			}
		}
	}


	template<typename TDataType>
	void MujocoXMLLoader<TDataType>::resetStates() 
	{

		this->createRigidBodySystem();


		ArticulatedBody<TDataType>::resetStates();

	}





	DEFINE_CLASS(MujocoXMLLoader);
}
