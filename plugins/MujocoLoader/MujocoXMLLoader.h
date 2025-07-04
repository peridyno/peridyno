/**
 * Copyright 2022 Yuzhong Guo
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
#include "Node/ParametricModel.h"

#include "tinyxml/tinyxml2.h"

#include "RigidBody/ArticulatedBody.h"

#include <optional>

namespace dyno
{
	struct XMLClass;

	struct XMLCommon 
	{	
		std::optional<std::string> name;
		std::optional<int> group;
		std::optional<Vec3f> size;
		std::optional<Vec4f> rgba;
		std::optional<Vec3f> pos;
		std::optional<Quat<Real>> quat;

		virtual void assign(const XMLCommon& other) 
		{
			if (!this->group.has_value() && other.group.has_value())
				this->group = other.group;

			if (!this->size.has_value() && other.size.has_value())
				this->size = other.size;

			if (!this->rgba.has_value() && other.rgba.has_value())
				this->rgba = other.rgba;

			if (!this->pos.has_value() && other.pos.has_value())
				this->pos = other.pos;

			if (!this->quat.has_value() && other.quat.has_value())
				this->quat = other.quat;
		}
	};

	struct XMLMesh
	{
		std::optional<std::string> name;
		std::optional<std::string> file;
		std::vector<Vec3f> vertices;
		std::vector<TopologyModule::Triangle> triangles;

	};


	//plane, hfield, sphere, capsule, ellipsoid, cylinder, box, mesh, sdf
	enum XMLGeomType
	{
		plane = 0,
		hfield = 1,
		sphere = 2,
		capsule = 3,
		ellipsoid = 4,
		cylinder = 5,
		box = 6,
		mesh = 7,
		sdf = 8
	};
	struct XMLSite :XMLCommon
	{
		std::optional<XMLGeomType> type = XMLGeomType::sphere;
		std::optional<std::string> meshName ;

		void assign(const XMLSite& other)
		{
			XMLCommon::assign(other);

			if (!this->type.has_value() && other.type.has_value())
				this->type = other.type;

			if (!this->meshName.has_value() && other.meshName.has_value())
				this->meshName = other.meshName;
		}

	};
	struct XMLGeom :XMLSite
	{
		std::optional<float> mass;
		std::optional<float> density;

		std::optional<int> contype;
		std::optional<int>  conaffinity;

		std::optional<Vec3f> friction;
		std::shared_ptr<XMLClass> XmlClass = nullptr;

		void assign(const XMLGeom& other) {
			
			XMLSite::assign(other);

			if (!this->mass.has_value() && other.mass.has_value())
				this->mass = other.mass;

			if (!this->density.has_value() && other.density.has_value())
				this->density = other.density;

			if (!this->contype.has_value() && other.contype.has_value())
				this->contype = other.contype;

			if (!this->conaffinity.has_value() && other.conaffinity.has_value())
				this->conaffinity = other.conaffinity;

			if (!this->friction.has_value() && other.friction.has_value())
				this->friction = other.friction;
		}
	};

	//free, ball, slide, hinge

	enum XMLJointType
	{
		free = 0,
		ball = 1,
		slide = 2,
		hinge = 3
	};

	struct XMLJoint
	{
		std::optional<std::string> name;
		std::optional<XMLJointType> type = XMLJointType::hinge;
		std::optional<float> damping;
		std::optional<Vec3f> axis;
		std::optional<Vec2f> range ;

		std::shared_ptr<XMLClass> XmlClass = nullptr;

		void assign(const XMLJoint& other) {

			if (!this->name.has_value() && other.name.has_value())
				this->name = other.name;

			if (!this->type.has_value() && other.type.has_value())
				this->type = other.type;

			if (!this->damping.has_value() && other.damping.has_value())
				this->damping = other.damping;

			if (!this->axis.has_value() && other.axis.has_value())
				this->axis = other.axis;

			if (!this->range.has_value() && other.range.has_value())
				this->range = other.range;

		}
	};

	struct XMLClass 
	{
		std::shared_ptr<XMLClass> parent = nullptr;
		std::optional<std::string> name;
		
		std::shared_ptr<XMLGeom> geomData = nullptr;
		std::shared_ptr<XMLJoint> jointData = nullptr;
		std::shared_ptr<XMLSite> siteData = nullptr;
	};


	struct XMLBody : XMLCommon
	{
		std::optional<Vec3f> inertialPos;
		std::shared_ptr<XMLClass> childClass;
		std::vector<std::shared_ptr<XMLGeom>> bodyGeoms;
		std::shared_ptr<XMLBody> parentBody = nullptr;
		std::optional<float> mass;
		std::shared_ptr<XMLJoint> joint;
		std::optional<Vec3f> realPos;
		std::optional<Vec3f> massCenter;

		int shapeId = -1;
		std::shared_ptr<PdActor> actor = nullptr;
	};

	struct XMLWorldBody : XMLCommon
	{
		std::shared_ptr<XMLClass> childClass;
	};

	struct XMLCompiler
	{
		std::string angle = "radian";
		std::string meshdir = "assets";
		bool autolimits = true;
	};

	struct Mesh
	{
		Mesh() {}
		Mesh(const std::vector<Vec3f>& v, const std::vector<TopologyModule::Triangle>& t)
		{
			this->vertices = v;
			this->triangles = t;
		}

		Mesh(const Mesh& mesh)
		{
			this->vertices = mesh.vertices;
			this->triangles = mesh.triangles;
		}

		std::vector<Vec3f> vertices;
		std::vector<TopologyModule::Triangle> triangles;
	};

	template<typename TDataType>
	class MujocoXMLLoader : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(MujocoXMLLoader, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename TopologyModule::Triangle Triangle;


		MujocoXMLLoader();
		~MujocoXMLLoader();

		std::string getNodeType() override { return "IO"; }

	public:

		DEF_VAR(int, Num,1, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");


	protected:
		void resetStates() override;

		void createRigidBodySystem();


	public:



	private:

		Vec4f decodeVec4f(const std::string str)
		{
			std::stringstream ss(str);
			std::string substr;

			ss >> substr;
			float x = std::stof(substr.c_str());

			ss >> substr;
			float y = std::stof(substr.c_str());

			ss >> substr;
			float z = std::stof(substr.c_str());

			ss >> substr;
			float w = std::stof(substr.c_str());

			return Vec4f(x, y, z, w);
		}

		std::string getFileExtension(const std::string& filename) {
			size_t pos = filename.rfind('.');
			if (pos == std::string::npos) {
				return "";
			}
			return filename.substr(pos);
		}

		Vec4f ParseVec4f(const char* rgbaStr) {
			Vec4f rgba;
			if (!rgbaStr) return rgba;

			std::istringstream iss(rgbaStr);
			float val;
			int index = 0;
			while (iss >> val) {
				rgba[index] = val;
				index++;
			}

			return rgba;
		}

		Vec3f ParseVec3f(const char* vec3fStr) {
			Vec3f vec3;
			if (!vec3fStr) return vec3;

			std::istringstream iss(vec3fStr);
			float val;
			int index = 0;
			while (iss >> val) {
				vec3[index] = val;
				index++;
			}
			return vec3;
		}

		Vec3f convertCoord(const Vec3f p) 
		{
			Mat3f rot = Mat3f(1, 0, 0,
				0, 0, 1,
				0, -1, 0);

			return rot * p;
		}

		Quat<Real> convertQuat(const Quat<Real> q)
		{
			
			Quat<Real> temp = Quat<Real>(q.y, q.z, q.w, q.x);
			temp.normalize();
			Mat3f R_mujoco = temp.toMatrix3x3();

			Mat3f M = Mat3f(1, 0, 0,
				0, 0, 1,
				0, -1, 0);

			Mat3f R = M * R_mujoco * M.transpose();

			return Quat<Real>(R);

		

		}

		int countFields(const char* attrValue) {
			if (!attrValue) return 0;  

			char* temp = new char[strlen(attrValue) + 1];
			strcpy(temp, attrValue);

			int count = 0;
			const char* delim = " \t\n\r";  
			char* token = strtok(temp, delim);
			while (token != nullptr) {
				++count;
				token = strtok(nullptr, delim);
			}

			delete[] temp;
			return count;
		}

		void convertVertices(std::vector<Vec3f>& vertices)
		{
			for (size_t i = 0; i < vertices.size(); i++)
			{
				vertices[i] = convertCoord(vertices[i]);
			}
		}

		Vec2f ParseVec2f(const char* vec2fStr) {
			Vec2f vec2;
			if (!vec2fStr) return vec2;

			std::istringstream iss(vec2fStr);
			float val;
			int index = 0;
			while (iss >> val) {
				vec2[index] = val;
				index++;
			}
			return vec2;
		}

		void parseGeomElemet(tinyxml2::XMLElement* geomElement, std::shared_ptr<XMLGeom> geom);
		void coutGeomElemet(std::shared_ptr<XMLGeom> geom);
		void parseSiteElemet(tinyxml2::XMLElement* geomElement, std::shared_ptr<XMLSite> site);
		void parseJointElemet(tinyxml2::XMLElement* jointElement, std::shared_ptr<XMLJoint> joint);
		void parseDefaultElement(tinyxml2::XMLElement* element, int depth = 0, std::shared_ptr<XMLClass> parent = nullptr);
		void parseCompilerElement(tinyxml2::XMLElement* element);

		void parseWorldBodyElement(tinyxml2::XMLElement* element);

		void parseAsset(tinyxml2::XMLElement* assetElement,const std::string& assetPath);

		void parseBodyElement(tinyxml2::XMLElement* element, int depth = 0, std::shared_ptr<XMLBody> parent = nullptr);
		bool filePathChanged();
		void updateTransform();
		Mesh mergeMesh(const Mesh& a, const Mesh& b);
		Mesh getMeshByName(std::string name);

		void offsetTriangleIndex(std::vector<TopologyModule::Triangle>& triangles, const int& offset);
		void offsetVertices(std::vector<Vec3f>& vertices, const Vec3f& offset);
		void rotateVertices(std::vector<Vec3f>& vertices, Quat<Real> q);

		void parseJointFromXML(const XMLClass& xmlClass,std::shared_ptr<XMLJoint> joint)
		{

			if (xmlClass.jointData)
				joint->assign(*xmlClass.jointData.get());

			if (xmlClass.parent)
				parseJointFromXML(*xmlClass.parent,joint);
			
		}

		void parseGeomFromXML(const XMLClass& xmlClass, std::shared_ptr<XMLGeom> geom)
		{

			if (xmlClass.jointData)
				geom->assign(*xmlClass.geomData.get());

			if (xmlClass.parent)
				parseGeomFromXML(*xmlClass.parent, geom);

		}

		std::shared_ptr<XMLClass> findClass(std::shared_ptr<XMLClass> xmlClass, std::string name)
		{
			if (xmlClass->name == name)
				return xmlClass;
			else if (xmlClass->parent)
				return findClass(xmlClass->parent, name);
			else
				return nullptr;		
			
		}

		bool isCollision(const XMLGeom& geom);


	private:

		std::vector<XMLMesh> mXMLAssetMeshes;
		std::vector<std::shared_ptr<XMLClass>> mXMLDefaultClass;
		std::vector<std::shared_ptr<XMLBody>> mXMLBody;
		std::vector<Vec3f> initialShapeCenter;
		XMLCompiler complier;

	};



	IMPLEMENT_TCLASS(MujocoXMLLoader, TDataType);
}