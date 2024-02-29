#include "GltfLoader.h"
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tinygltf/tiny_gltf.h"


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
		for (size_t i = 0; i < meshNum; i++)
		{
			printf("model size = %d \n", meshNum);

			int primInfoNum = model.meshes[i].primitives.size();
			for (size_t j = 0; j < primInfoNum; j++)
			{
				printf("model: %d, primtive: %d\n",i,j);
				//====================================
				//current primitive
				const tinygltf::Primitive& primitive = model.meshes[i].primitives[j];
				printf("triangle Mode %d\n", primitive.mode);
				std::map<std::string, int> a = primitive.attributes;

				//triangle accessor
				{
					const tinygltf::Accessor& accessorTriangles = model.accessors[primitive.indices];
					const tinygltf::BufferView& bufferView = model.bufferViews[accessorTriangles.bufferView];
					const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

					std::vector<TopologyModule::Triangle> triangles;
					//get Triangle vertex id
					if (accessorTriangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
					{
						printf("\n----------   UNSIGNED_BYTE   ---------\n");
						const byte* elementBYTE = reinterpret_cast<const byte*>(&buffer.data[accessorTriangles.byteOffset + model.accessors[0].byteOffset]);

						//for (size_t k = 0; k < accessorTriangles.count / 3; k++)
						//{
						//	std::cout << int(elementBYTE[k * 3]) <<", " << int(elementBYTE[k * 3 + 1]) << ", " << int(elementBYTE[k * 3 + 2]) << ",";
						//	printf("\n");
						//	triangles.push_back(TopologyModule::Triangle(int(elementBYTE[k * 3]), int(elementBYTE[k * 3 + 1]), int(elementBYTE[k * 3 + 2])));
						//}
						//triangleSet->setTriangles(triangles);

						for (size_t k = 0; k < accessorTriangles.count; k++)
						{
							auto num = this->varTest()->getValue();
							std::cout << int(elementBYTE[k]) % num << ","; //
							if ((k + 1) % 3 == 0) 
							{
								std::cout << " -- " << int(elementBYTE[k]) / num;
								printf("\n");
							}
								
						}


					}
					else if (accessorTriangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
					{
						printf("\n----------   UNSIGNED_SHORT   ---------\n");
						const unsigned short* elements = reinterpret_cast<const unsigned short*>(&buffer.data[accessorTriangles.byteOffset + model.accessors[0].byteOffset]);

						for (size_t k = 0; k < accessorTriangles.count / 3; k++)
						{
							std::cout << elements[k] << ",";
							if ((k + 1) % 3 == 0)
								printf("\n");
						}
					}
					else if (accessorTriangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
					{
						printf("\n----------   UNSIGNED_INT   ---------\n");
						const unsigned int* elements = reinterpret_cast<const unsigned int*>(&buffer.data[accessorTriangles.byteOffset + model.accessors[0].byteOffset]);

						for (size_t k = 0; k < accessorTriangles.count / 3; k++)
						{
							std::cout << elements[k] << ",";
							if ((k + 1) % 3 == 0)
								printf("\n");
						}
					}
				}

				//assign Attributes for Points
				for (std::map<std::string, int>::iterator it = a.begin(); it != a.end(); ++it)
				{
					std::cout << "***find Parm !!! " << it->first << std::endl;
					std::string parm= it->first;



					//Check AttributeName
					if(it->first == "POSITION")
					{
						printf("\n--------------------  POSITION  --------------------\n");
						const tinygltf::Accessor& accessorPosition = model.accessors[primitive.attributes.find("POSITION")->second];
						const tinygltf::BufferView& bufferView = model.bufferViews[accessorPosition.bufferView];
						const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
						CArray<Coord> c_P;
						DArray<Coord> d_P;
						const float* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessorPosition.byteOffset]);
						for (size_t i = 0; i < accessorPosition.count; ++i) 
						{
							c_P.pushBack(Coord(positions[i * 3 + 0],positions[i * 3 + 1], positions[i * 3 + 2]));
						}

						//d_P.assign(c_P);
						//this->statePosition()->getDataPtr()->assign(d_P);;
						//triangleSet->setPoints(d_P);

						printf("Position.size : %d",this->statePosition()->size());
					}
				}
				

				




			}
		}

	}


	DEFINE_CLASS(GltfLoader);
}