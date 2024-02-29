#include "GltfLoader.h"



#include <bitset>
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"



namespace dyno
{

	template<typename TDataType>
	GltfLoader<TDataType>::GltfLoader()
	{
		auto callback = std::make_shared<FCallBackFunc>(std::bind(&GltfLoader<TDataType>::varChanged, this));

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<DataType3f>>());

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varFileName()->attach(callback);
		this->varRealName_1()->attach(callback);
		this->varRealName_2()->attach(callback);
		this->varCoordName_1()->attach(callback);
		this->varCoordName_2()->attach(callback);

		auto tsRender = std::make_shared<GLSurfaceVisualModule>();
		tsRender->setColor(Color(0.8f, 0.52f, 0.25f));
		tsRender->setVisible(true);
		this->stateTriangleSet()->connect(tsRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(tsRender);

		auto esRender = std::make_shared<GLWireframeVisualModule>();
		esRender->varBaseColor()->setValue(Color(0, 0, 0));
		this->stateTriangleSet()->connect(esRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(esRender);

		this->stateTriangleSet()->promoteOuput();

		this->varTest()->setRange(0,500);
	}

	template<typename TDataType>
	void GltfLoader<TDataType>::varChanged()
	{
		if (this->varFileName()->isEmpty())
			return;


		auto triangleSet = this->stateTriangleSet()->getDataPtr();

		using namespace tinygltf;

		Model model;
		TinyGLTF loader;
		std::string err;
		std::string warn;
		std::string filename = this->varFileName()->getValue().string();


		bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
		//bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, argv[1]); // for binary glTF(.glb)

		if (!warn.empty()) {
			printf("Warn: %s\n", warn.c_str());
		}

		if (!err.empty()) {
			printf("Err: %s\n", err.c_str());
		}

		if (!ret) {
			printf("Failed to parse glTF\n");
			return;
		}

		//model.meshes[0].primitives[0].attributes;

		int meshNum = model.meshes.size();
		printf("**** mesh number = %d \n", meshNum);

		for (size_t i = 0; i < meshNum; i++)
		{	
			
			int primNum = model.meshes[i].primitives.size();
			printf("**** Primtive Number : %d\n", primNum);

			for (size_t j = 0; j < primNum; j++)
			{
				printf("get Data from mesh: %d, primtive: %d\n",i,j);

				//current primitive
				const tinygltf::Primitive& primitive = model.meshes[i].primitives[j];

				std::map<std::string, int> a = primitive.attributes;
				
				//Print Attributes;
				for (std::map<std::string, int>::iterator it = a.begin(); it != a.end(); ++it)
				{
					std::cout << "***find Parm !!! " << it->first << std::endl;
					std::string parm = it->first;
				} 

				//Set Vertices
				std::vector<Coord> vertices;
				this->getCoordByAttributeName(model, primitive, std::string("POSITION"), vertices);
				triangleSet->setPoints(vertices);
				vertices.clear();

				//Set Normal
				std::vector<Coord> normals;
				this->getCoordByAttributeName(model, primitive, std::string("NORMAL"), normals);
				this->stateNormal()->assign(normals);
				normals.clear();

				//Set TexCoord
				std::vector<Coord> texCoord0;
				this->getCoordByAttributeName(model, primitive, std::string("TEXCOORD_0"), normals);
				this->stateTexCoord_0()->assign(texCoord0);
				texCoord0.clear();

				std::vector<Coord> texCoord1;
				this->getCoordByAttributeName(model, primitive, std::string("TEXCOORD_0"), normals);
				this->stateTexCoord_0()->assign(texCoord1);
				texCoord1.clear();


				//Set Triangles
				if (primitive.mode == TINYGLTF_MODE_TRIANGLES)
				{
					printf("*********** getTriangles ,Mode ==  TINYGLTF_MODE_TRIANGLES  ************");

					std::vector<TopologyModule::Triangle> trianglesVector;
					getTrianglesFromMesh(model, primitive,trianglesVector);
					triangleSet->setTriangles(trianglesVector);
					trianglesVector.clear();
				}

				
			}
		}

		printf("*********\n*********\n*********\n*********\n*********\n");
	}

	// ***************************** function *************************** //
	template<typename TDataType>
	void GltfLoader<TDataType>::getTrianglesFromMesh(
		tinygltf::Model& model, 
		const tinygltf::Primitive& primitive,
		std::vector<TopologyModule::Triangle>& triangles
	)
	{	
			const tinygltf::Accessor& accessorTriangles = model.accessors[primitive.indices];
			const tinygltf::BufferView& bufferView = model.bufferViews[accessorTriangles.bufferView];
			const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
			
			//get Triangle Vertex id
			if (accessorTriangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
			{
				printf("\n----------   UNSIGNED_BYTE   ---------\n");
				const byte* elements = reinterpret_cast<const byte*>(&buffer.data[accessorTriangles.byteOffset + model.accessors[0].byteOffset]);

				for (size_t k = 0; k < accessorTriangles.count / 3; k++)
				{
					std::cout << int(elements[k * 3]) << ", " << int(elements[k * 3 + 1]) << ", " << int(elements[k * 3 + 2]) << ",";
					printf("\n");
					triangles.push_back(TopologyModule::Triangle(int(elements[k * 3]), int(elements[k * 3 + 1]), int(elements[k * 3 + 2])));
				}

			}
			else if (accessorTriangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
			{
				printf("\n----------   UNSIGNED_SHORT   ---------\n");
				const unsigned short* elements = reinterpret_cast<const unsigned short*>(&buffer.data[accessorTriangles.byteOffset + model.accessors[0].byteOffset]);

				for (size_t k = 0; k < accessorTriangles.count / 3; k++)
				{
					std::cout << int(elements[k * 3]) << ", " << int(elements[k * 3 + 1]) << ", " << int(elements[k * 3 + 2]) << ",";
					printf("\n");
					triangles.push_back(TopologyModule::Triangle(int(elements[k * 3]), int(elements[k * 3 + 1]), int(elements[k * 3 + 2])));
				}
				
			}
			else if (accessorTriangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
			{
				printf("\n----------   UNSIGNED_INT   ---------\n");
				const unsigned int* elements = reinterpret_cast<const unsigned int*>(&buffer.data[accessorTriangles.byteOffset + model.accessors[0].byteOffset]);

				for (size_t k = 0; k < accessorTriangles.count / 3; k++)
				{
					std::cout << int(elements[k * 3]) << ", " << int(elements[k * 3 + 1]) << ", " << int(elements[k * 3 + 2]) << ",";
					printf("\n");
					triangles.push_back(TopologyModule::Triangle(int(elements[k * 3]), int(elements[k * 3 + 1]), int(elements[k * 3 + 2])));
				}
				
			}
	}

	// ********************************** getVec3f By Attribute Name *************************//
	template<typename TDataType>
	void GltfLoader<TDataType>::getCoordByAttributeName(
		tinygltf::Model& model,
		const tinygltf::Primitive& primitive,
		std::string& attributeName,
		std::vector<Coord>& vertices
		)
	{
		//assign Attributes for Points
		std::map<std::string, int>::const_iterator iter;
		iter = primitive.attributes.find(attributeName);

		if (iter == primitive.attributes.end())
		{	
			std::cout << attributeName << " : not found !!! \n";
		}

		const tinygltf::Accessor& accessorAttribute = model.accessors[iter->second];
		const tinygltf::BufferView& bufferView = model.bufferViews[accessorAttribute.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
		std::cout << attributeName <<"Type == " << accessorAttribute.type << "\n";
		
		if (accessorAttribute.type == TINYGLTF_TYPE_VEC3) 
		{
			const float* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
			for (size_t i = 0; i < accessorAttribute.count; ++i)
			{
				vertices.push_back(Coord(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]));
				std::cout << i << "-------" << positions[i * 3 + 0] << ", " << positions[i * 3 + 1] << ", " << positions[i * 3 + 2] << "\n";
			}
		}
		else if (accessorAttribute.type == TINYGLTF_TYPE_VEC2)
		{
			const float* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
			for (size_t i = 0; i < accessorAttribute.count; ++i)
			{
				vertices.push_back(Coord(positions[i * 2 + 0], positions[i * 2 + 1], 0));
				std::cout << i << "-------" << positions[i * 2 + 0] << ", " << positions[i * 2 + 1] << ", " << 0 << "\n";
			}
		}

	}




	DEFINE_CLASS(GltfLoader);
}