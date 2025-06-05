#include "Gear.h"

#include "Mapping/DiscreteElementsToTriangleSet.h"
#include "GLSurfaceVisualModule.h"
#include <cstddef>
#include <string>

namespace dyno
{
	//Gear
	IMPLEMENT_TCLASS(Gear, TDataType)

	template<typename TDataType>
	Gear<TDataType>::Gear() :
		ArticulatedBody<TDataType>()
	{
		
		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		this->stateTopology()->connect(mapper->inDiscreteElements());
		this->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->setColor(Color(1, 1, 0));
		sRender->setAlpha(0.5f);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);
		
	}

	template<typename TDataType>
	Gear<TDataType>::~Gear()
	{

	}

	template<typename TDataType>
	void Gear<TDataType>::resetStates()
	{	
		this->clearRigidBodySystem();
		this->clearVechicle();

		std::string filename = getAssetPath() + "gear/gear_up.obj";
		if (this->varFilePath()->getValue() != filename)
		{
			this->varFilePath()->setValue(FilePath(filename));
		}

		//first gear
		RigidBodyInfo info;
		info.position = Vec3f(0, 1, 0);
		info.angularVelocity = Vec3f(1, 0, 0);
		info.motionType = BodyType::Kinematic;
		info.bodyId = 0;
		auto actor = this->createRigidBody(info);

		CapsuleInfo capsule;
		capsule.rot = Quat1f(M_PI / 2, Vec3f(0, 0, 1));
		capsule.radius = 0.05f;
		capsule.halfLength = 0.26f;

		float r = 0.798f;
		for (uint sec = 0; sec < 24; sec++)
		{
			float theta = sec * M_PI / 12 + 0.115;
			float y = r * sin(theta);
			float z = r * cos(theta);

			capsule.center = Vec3f(-0.042f, y, z);
			this->bindCapsule(actor, capsule);
		}

		this->bindShape(actor, Pair<uint, uint>(0, 0));

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}

	DEFINE_CLASS(Gear);


	IMPLEMENT_TCLASS(Bug, TDataType)

	template<typename TDataType>
	Bug<TDataType>::Bug() :
		ArticulatedBody<TDataType>()
	{	
		/*
		auto mapper = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
		this->stateTopology()->connect(mapper->inDiscreteElements());
		this->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->setColor(Color(1, 1, 0));
		sRender->setAlpha(0.5f);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);*/
		this->varDoTransform()->setValue(false);
		
	}

	template<typename TDataType>
	Bug<TDataType>::~Bug()
	{

	}

	template<typename TDataType>
	void Bug<TDataType>::resetStates()
	{
		this->clearRigidBodySystem();
		this->clearVechicle();
		this->vertices.clear();
		this->edges.clear();
		this->faces.clear();
		std::string model(this->file_name);
		this->loadMa(model);
		std::string filename = getAssetPath() + model + std::string(".obj");
		if (this->varFilePath()->getValue() != filename)
		{
			this->varFilePath()->setValue(FilePath(filename));
		}

		RigidBodyInfo info;
		//info.angle = Quat1f(M_PI / 2, Vec3f(1, 0, 0));
		MedialConeInfo medalcone;
		MedialSlabInfo medalslab;
		info.position = Vec3f(0) + this->position;
		info.linearVelocity = this->velocity;
		info.angularVelocity = Vec3f(-2, 0, 0);
		info.motionType = BodyType::Dynamic;
		info.bodyId = 0;
		auto actor = this->createRigidBody(info);


		for(size_t i = 0;i < edges.size();i++)
		{
			Vec2i edge = edges[i];
			if(edge[0] >= vertices.size() || edge[1] >= vertices.size())
			{
				std::cerr << "ERROR load edge" << std::endl;
				continue;
			}
			medalcone.v[0] = Vec3f(vertices[edge[0]][0], vertices[edge[0]][1], vertices[edge[0]][2]);
			medalcone.v[1] = Vec3f(vertices[edge[1]][0], vertices[edge[1]][1], vertices[edge[1]][2]);
			medalcone.radius[0] = vertices[edge[0]][3];
			medalcone.radius[1] = vertices[edge[1]][3];
			this->bindMedialCone(actor, medalcone, 1000);
		}

		for(size_t i = 0;i < faces.size();i++)
		{
			Vec3i face = faces[i];
			if(face[0] >= vertices.size() || face[1] >= vertices.size() || face[2] >= vertices.size())
			{
				std::cerr << "ERROR load face" << std::endl;
				continue;
			}
			medalslab.v[0] = Vec3f(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2]);
			medalslab.v[1] = Vec3f(vertices[face[1]][0], vertices[face[1]][1], vertices[face[1]][2]);
			medalslab.v[2] = Vec3f(vertices[face[2]][0], vertices[face[2]][1], vertices[face[2]][2]);
			medalslab.radius[0] = vertices[face[0]][3];
			medalslab.radius[1] = vertices[face[1]][3];
			medalslab.radius[2] = vertices[face[2]][3];
			this->bindMedialSlab(actor, medalslab, 1000);
		}

		this->bindShape(actor, Pair<uint, uint>(0, 0));

		//**************************************************//
		ArticulatedBody<TDataType>::resetStates();
	}

	template<typename TDataType>
	void Bug<TDataType>::loadMa(std::string file_path)
	{
		std::string filename = getAssetPath() + file_path + std::string(".ma");
		std::ifstream inputFile(filename);
		std::string line;
		int num_vertices = 0, num_edges = 0, num_faces = 0;

		if (std::getline(inputFile, line))
		{
			std::istringstream iss(line);
			if (!(iss >> num_vertices >> num_edges >> num_faces)) {
				std::cerr << "ERROR MA FILE" << std::endl;
				inputFile.close();
			}
			std::cout << "num Of vertices : " << num_vertices
				<< ", num Of edges : " << num_edges
				<< ", num Of faces : " << num_faces << std::endl;
		}
		else
		{
			std::cerr << "ERROR MA FILE" << std::endl;
			inputFile.close();
		}

		int current_vertex_count = 0;
		int current_edge_count = 0;
		int current_face_count = 0;

		while(std::getline(inputFile, line))
		{
			if(line.empty() || line[0] == '#')
			{
				continue;
			}

			std::istringstream iss(line);
			char type;
			iss >> type;

			if(type == 'v' && current_vertex_count < num_vertices)
			{
				Vec4f vertex;
				if(iss >> vertex[0] >> vertex[1] >> vertex[2] >> vertex[3])
				{
					vertex = vertex;
					vertices.push_back(vertex);
					current_vertex_count++;
				}
				else{
					std::cerr << "ERROR load vertex" << std::endl;
				}
			}
			else if(type == 'e' && current_edge_count < num_edges)
			{
				Vec2i edge;
				if(iss >> edge[0] >> edge[1])
				{
					if(edge[0] >= 0 && edge[0] < num_vertices && edge[1] >= 0 && edge[1] < num_vertices)
					{
						edges.push_back(edge);
						current_edge_count++;
					}
					else {
						std::cerr << "ERROR load edge" << std::endl;
					}
				}
				else{
					std::cerr << "ERROR load edge" << std::endl;
				}
			}
			else if(type == 'f' && current_face_count < num_faces)
			{
				Vec3i face;
				if(iss >> face[0] >> face[1] >> face[2])
				{
					if(face[0] >= 0 && face[0] < num_vertices && face[1] >= 0 && face[1] < num_vertices && face[2] >= 0 && face[2] < num_vertices)
					{
						faces.push_back(face);
						current_face_count++;
					}
					else {
						std::cerr << "ERROR load face" << std::endl;
					}
				}
				else{
					std::cerr << "ERROR load face" << std::endl;
				}
			}
		}
		inputFile.close();
		
	}

	DEFINE_CLASS(Bug);
}
