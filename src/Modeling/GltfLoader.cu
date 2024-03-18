#include "GltfLoader.h"

#include "GLPointVisualModule.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

#include <GLPhotorealisticRender.h>

namespace dyno
{
	bool loadImage(const char* path, dyno::CArray2D<dyno::Vec4f>& img)
	{
		int x, y, comp;
		stbi_set_flip_vertically_on_load(true);

		float* data = stbi_loadf(path, &x, &y, &comp, STBI_default);

		if (data) {
			img.resize(x, y);
			for (int x0 = 0; x0 < x; x0++)
			{
				for (int y0 = 0; y0 < y; y0++)
				{
					int idx = (y0 * x + x0) * comp;
					for (int c0 = 0; c0 < comp; c0++) {
						img(x0, y0)[c0] = data[idx + c0];
					}
				}
			}
			STBI_FREE(data);
		}

		return data != 0;
	}



	template<typename TDataType>
	GltfLoader<TDataType>::GltfLoader()
	{
		auto callback = std::make_shared<FCallBackFunc>(std::bind(&GltfLoader<TDataType>::varChanged, this));


		this->stateJointSet()->setDataPtr(std::make_shared<EdgeSet<DataType3f>>());

		this->varImportAnimation()->attach(callback);
		this->varFileName()->attach(callback);

		auto render = std::make_shared<GLPhotorealisticRender>();

		this->stateVertex()->connect(render->inVertex());
		this->stateNormal()->connect(render->inNormal());
		this->stateTexCoord_0()->connect(render->inTexCoord());

		this->stateShapes()->connect(render->inShapes());
		this->stateMaterials()->connect(render->inMaterials());
		this->graphicsPipeline()->pushModule(render);


		//Joint Render
		auto jointRender = std::make_shared<GLPointVisualModule>();
		jointRender->setColor(Color(1.0f, 0.0f, 0.0f));
		jointRender->varPointSize()->setValue(this->varJointRadius()->getValue());
		jointRender->setVisible(true);
		this->stateJointSet()->connect(jointRender->inPointSet());
		this->graphicsPipeline()->pushModule(jointRender);

		auto jointLineRender = std::make_shared<GLWireframeVisualModule>();
		jointLineRender->varBaseColor()->setValue(Color(0, 1, 0));
		jointLineRender->setVisible(true);
		jointLineRender->varRadius()->setValue(this->varJointRadius()->getValue() / 2);
		jointLineRender->varRenderMode()->setCurrentKey(GLWireframeVisualModule::EEdgeMode::CYLINDER);
		this->stateJointSet()->connect(jointLineRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(jointLineRender);


		this->stateVertex()->promoteOuput();
		this->stateNormal()->promoteOuput();

		this->stateShapes()->promoteOuput();
		this->stateMaterials()->promoteOuput();



		this->stateShapes()->promoteOuput();
	}

	template<typename TDataType>
	void GltfLoader<TDataType>::varChanged()
	{
		if (this->varFileName()->isEmpty())
			return;

		printf("!!!!!!!!!!!!!!!!!    Import GLTF   !!!!!!!!!!!!!!!!!!!!!!!!\n\n\n");

		this->InitializationData();


		using namespace tinygltf;

		auto newModel = new Model;
		model = *newModel;
		delete newModel;

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


		////import Scenes :


		////import Animation ：
		if (this->varImportAnimation()->getValue()) 
		{
			this->importAnimation();
		}
		



		//Animation done;

		std::vector<std::vector<int>> Scene_Nodes;

		std::map<scene, std::vector<int>> Scene_JointsNodesId;
		std::map<scene, std::vector<int>> Scene_SkinNodesId;
		std::map<scene, std::vector<int>> Scene_CamerasNodesId;

		// import Scenes:
		for (size_t i = 0; i < model.scenes.size(); i++)
		{
			Scene_Name.push_back(model.scenes[i].name);
			Scene_Nodes.push_back(model.scenes[i].nodes);

			std::vector<joint> vecS_Joints;
			std::vector<joint> vecS_Meshs;
			std::vector<joint> vecS_Cameras;
			std::vector<joint> vecS_Skins;
			for (size_t n = 0; n < model.scenes[i].nodes.size(); n++)
			{
				//遍历场景内的节点
				int nId = model.scenes[i].nodes[n];
				if (model.nodes[nId].mesh == -1 && model.nodes[nId].camera == -1)
				{
					vecS_Joints.push_back(nId);
				}
				if (model.nodes[nId].mesh != -1)
				{
					vecS_Meshs.push_back(nId);
				}
				if (model.nodes[nId].camera != -1)
				{
					vecS_Cameras.push_back(nId);
				}
				if (model.nodes[nId].skin != -1)
				{
					vecS_Skins.push_back(nId);
				}
			}
			Scene_JointsNodesId[i] = vecS_Joints;

			Scene_CamerasNodesId[i] = vecS_Cameras;

		}



		this->getJointAndHierarchy(Scene_JointsNodesId, all_Joints);		//update private: std::map<joint, std::vector<int>> jointId_joint_Dir;

		int jointNum = all_Joints.size();



		// Joint import from root
		//joint

		std::vector<std::string> joint_name;
		std::vector<std::vector<int>> joint_child;

		//get Local Transform T S R M
		this->getJointsTransformData(all_Joints, joint_name, joint_child);

		//get InverseBindMatrix (Global)
		this->buildInverseBindMatrices(all_Joints);	//依赖层级

		// get joint World Location		这计算点位置和动画变换写在一起了
		printf("************  Set Joint  ************\n");
		{
			std::vector<Coord> jointVertices;
			for (size_t j = 0; j < jointNum; j++)
			{
				joint jId = all_Joints[j];

				if (this->varImportAnimation()->getValue())
				{

					this->updateAnimationMatrix(all_Joints, this->stateFrameNumber()->getValue());

					jointVertices.push_back(getVertexLocationWithJointTransform(jId, Vec3f(0, 0, 0), joint_AnimaMatrix));

				}
				else
				{
					jointVertices.push_back(getVertexLocationWithJointTransform(jId, Vec3f(0, 0, 0), joint_matrix));
				}
			}

			//
			this->stateJointSet()->getDataPtr()->setPoints(jointVertices);
			std::vector<TopologyModule::Edge> edges;

			for (size_t jId = 0; jId < joint_child.size(); jId++)
			{
				for (auto childId : joint_child[jId])
				{
					edges.push_back(TopologyModule::Edge(jId, childId));
				}
			}
			this->stateJointSet()->getDataPtr()->setEdges(edges);
		}






		std::vector<Coord> vertices;
		std::vector<Coord> normals;
		std::vector<Coord> texCoord0;
		std::vector<Coord> texCoord1;



		std::vector<TopologyModule::Triangle> trianglesVector;
		int shapeNum = 0;

		for (auto meshId : model.meshes)
		{
			shapeNum += meshId.primitives.size();
		}


		//shape
		this->stateShapes()->resize(shapeNum);
		auto& statShapes = this->stateShapes()->getData();

		dyno::CArray2D<dyno::Vec4f> texture(1, 1);
		texture[0, 0] = dyno::Vec4f(1);

		this->stateMaterials()->resize(model.materials.size());
		auto& sMats = this->stateMaterials()->getData();



		std::vector<tinygltf::Material>& materials = model.materials;
		std::vector<Texture>& textures = model.textures;
		std::vector<Image>& images = model.images;



		int primitive_PointOffest;
		int currentShape = 0;

		for (int mId = 0; mId < model.meshes.size(); mId++)
		{
			// import Mesh

			std::vector<dyno::TopologyModule::Triangle> vertexIndex;
			std::vector<dyno::TopologyModule::Triangle> normalIndex;
			std::vector<dyno::TopologyModule::Triangle> texCoordIndex;

			int primNum = model.meshes[mId].primitives.size();

			for (size_t pId = 0; pId < primNum; pId++)	//shape
			{

				primitive_PointOffest = (vertices.size());

				//current primitive
				const tinygltf::Primitive& primitive = model.meshes[mId].primitives[pId];

				std::map<std::string, int> attributesName = primitive.attributes;


				//Set Vertices
				this->getCoordByAttributeName(model, primitive, std::string("POSITION"), vertices);

				//Set Normal
				this->getCoordByAttributeName(model, primitive, std::string("NORMAL"), normals);

				//Set TexCoord


				this->getCoordByAttributeName(model, primitive, std::string("TEXCOORD_0"), texCoord0);

				this->getCoordByAttributeName(model, primitive, std::string("TEXCOORD_1"), texCoord1);


				//Set Triangles

				if (primitive.mode == TINYGLTF_MODE_TRIANGLES)
				{
					//在多个场景时候潜在可能有问题，但没遇到过该类文件。

					std::vector<TopologyModule::Triangle> tempTriangles;

					getTriangles(model, primitive, tempTriangles, primitive_PointOffest);

					vertexIndex = (tempTriangles);
					normalIndex = (tempTriangles);
					texCoordIndex = (tempTriangles);

					statShapes[currentShape] = std::make_shared<Shape>();


					statShapes[currentShape]->vertexIndex.assign(vertexIndex);
					statShapes[currentShape]->normalIndex.assign(normalIndex);
					statShapes[currentShape]->texCoordIndex.assign(texCoordIndex);


					int matId = primitive.material;
					if (matId != -1)
					{

						auto material = materials[matId];
						auto color = material.pbrMetallicRoughness.baseColorFactor;
						auto roughness = material.pbrMetallicRoughness.roughnessFactor;

						auto metallic = material.pbrMetallicRoughness.metallicFactor;

						auto colorTexId = material.pbrMetallicRoughness.baseColorTexture.index;
						auto texCoord = material.pbrMetallicRoughness.baseColorTexture.texCoord;



						sMats[matId] = std::make_shared<Material>();
						sMats[matId]->ambient = { 0,0,0 };
						sMats[matId]->diffuse = Vec3f(color[0], color[1], color[2]);
						sMats[matId]->alpha = color[3];
						sMats[matId]->specular = Vec3f(1 - roughness);
						sMats[matId]->roughness = roughness;


						std::string colorUri = getTexUri(textures, images, colorTexId);

						if (!colorUri.empty())
						{
							auto root = this->varFileName()->getValue().parent_path();
							colorUri = (root / colorUri).string();

							if (loadImage(colorUri.c_str(), texture))
							{
								sMats[matId]->texColor.assign(texture);
							}
						}

						auto bumpTexId = material.normalTexture.index;
						auto scale = material.normalTexture.scale;
						std::string bumpUri = getTexUri(textures, images, bumpTexId);

						if (!bumpUri.empty())
						{
							auto root = this->varFileName()->getValue().parent_path();
							bumpUri = (root / bumpUri).string();

							if (loadImage(bumpUri.c_str(), texture))
							{
								sMats[matId]->texBump.assign(texture);
								sMats[matId]->bumpScale = scale;
							}
						}

						statShapes[currentShape]->material = sMats[matId];

					}



					trianglesVector.insert(trianglesVector.end(), tempTriangles.begin(), tempTriangles.end());

				}
				currentShape++;

			}
		}







		//Scene_SkinNodesId;
		////导入蒙皮

		std::vector<joint> skinJoints;
		for (auto skin : model.skins)
		{
			auto& joints = skin.joints;
			skinJoints.insert(skinJoints.end(), joints.begin(), joints.end());


		}


		for (int mId = 0; mId < model.meshes.size(); mId++)
		{
			// import Mesh
			int primNum = model.meshes[mId].primitives.size();


			for (size_t pId = 0; pId < primNum; pId++)
			{

				getVec4ByAttributeName(model, model.meshes[mId].primitives[pId], std::string("WEIGHTS_0"), meshVertex_joint_weight_0);//

				getVec4ByAttributeName(model, model.meshes[mId].primitives[pId], std::string("WEIGHTS_1"), meshVertex_joint_weight_1);//

				getVertexBindJoint(model, model.meshes[mId].primitives[pId], std::string("JOINTS_0"), meshVertex_bind_joint_0, skinJoints);

				getVertexBindJoint(model, model.meshes[mId].primitives[pId], std::string("JOINTS_1"), meshVertex_bind_joint_1, skinJoints);
			}
		}
		//replace JointId from Skin

//		int skeleton;             // The index of the node used as a skeleton root
//		std::vector<int> joints;  // Indices of skeleton nodes



		this->stateBindJoints_0()->assign(meshVertex_bind_joint_0);
		this->stateBindJoints_1()->assign(meshVertex_bind_joint_1);

		this->stateWeights_0()->assign(meshVertex_joint_weight_0);
		this->stateWeights_1()->assign(meshVertex_joint_weight_1);



		clock_t start_time = clock();

		//变换顶点
		//std::vector<Coord> aniVertices = vertices;

		this->stateVertex()->assign(vertices);

		initialPosition.assign(this->stateVertex()->getData());


		if (model.animations.size() && this->varImportAnimation()->getValue())
		{

			this->updateAnimationMatrix(all_Joints, this->stateFrameNumber()->getValue());
			this->updateJointWorldMatrix(all_Joints, joint_AnimaMatrix);

			cuExecute(this->stateVertex()->getData().size(),
				PointsAnimation,
				initialPosition,
				this->stateVertex()->getData(),
				this->stateJointInverseBindMatrix()->getData(),
				this->stateJointWorldMatrix()->getData(),

				this->stateBindJoints_0()->getData(),
				this->stateBindJoints_1()->getData(),
				this->stateWeights_0()->getData(),
				this->stateWeights_1()->getData()
			);

		}


		// assign Vertices Normal texCoord

		this->stateNormal()->assign(normals);

		std::vector<Vec2f> tempTexCoord;
		for (auto uv0 : texCoord0)
		{
			tempTexCoord.push_back(Vec2f(uv0[0], 1 - uv0[1]));// uv.v need flip
		}
		this->stateTexCoord_0()->assign(tempTexCoord);
		tempTexCoord.clear();
		for (auto uv1 : texCoord1)
		{
			tempTexCoord.push_back(Vec2f(uv1[0], 1 - uv1[1]));
		}
		this->stateTexCoord_1()->assign(tempTexCoord);
		texCoord1.clear();
		tempTexCoord.clear();



	}




	// ***************************** function *************************** //
	template<typename TDataType>
	void GltfLoader<TDataType>::traverseNode(joint id, std::vector<joint>& joint_nodes, std::map<joint, std::vector<int>>& dir, std::vector<joint> currentDir)
	{
		const tinygltf::Node& node = model.nodes[id];
		currentDir.push_back(id);
		joint_nodes.push_back(id);

		for (int childIndex : node.children) {
			const tinygltf::Node& childNode = model.nodes[childIndex];
			traverseNode(childIndex, joint_nodes, dir, currentDir);
		}

		std::reverse(currentDir.begin(), currentDir.end());	
		dir[id] = currentDir;
	}
	
	// ***************************** get triangle vertexID *************************** //
	template<typename TDataType>
	void GltfLoader<TDataType>::getTriangles(
		tinygltf::Model& model,
		const tinygltf::Primitive& primitive,
		std::vector<TopologyModule::Triangle>& triangles,
		int pointOffest
	)
	{
		const tinygltf::Accessor& accessorTriangles = model.accessors[primitive.indices];
		const tinygltf::BufferView& bufferView = model.bufferViews[accessorTriangles.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

		//get Triangle Vertex id
		if (accessorTriangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
		{
			const byte* elements = reinterpret_cast<const byte*>(&buffer.data[accessorTriangles.byteOffset + bufferView.byteOffset]);

			for (size_t k = 0; k < accessorTriangles.count / 3; k++)
			{
				triangles.push_back(TopologyModule::Triangle(int(elements[k * 3]) + pointOffest, int(elements[k * 3 + 1]) + pointOffest, int(elements[k * 3 + 2]) + pointOffest));
			}

		}
		else if (accessorTriangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
		{
			const unsigned short* elements = reinterpret_cast<const unsigned short*>(&buffer.data[accessorTriangles.byteOffset + bufferView.byteOffset]);

			for (size_t k = 0; k < accessorTriangles.count / 3; k++)
			{
				triangles.push_back(TopologyModule::Triangle(int(elements[k * 3]) + pointOffest, int(elements[k * 3 + 1]) + pointOffest, int(elements[k * 3 + 2]) + pointOffest));
			}

		}
		else if (accessorTriangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
		{
			const unsigned int* elements = reinterpret_cast<const unsigned int*>(&buffer.data[accessorTriangles.byteOffset + bufferView.byteOffset]);

			for (size_t k = 0; k < accessorTriangles.count / 3; k++)
			{
				triangles.push_back(TopologyModule::Triangle(int(elements[k * 3]) + pointOffest, int(elements[k * 3 + 1]) + pointOffest, int(elements[k * 3 + 2]) + pointOffest));
			}

		}
	}


	template<typename TDataType>
	void GltfLoader<TDataType>::getVec4ByAttributeName(tinygltf::Model& model,
		const tinygltf::Primitive& primitive,
		const std::string& attributeName,
		std::vector<Vec4f>& vec4Data
	)
	{
		//assign Attributes for Points
		std::map<std::string, int>::const_iterator iter;
		iter = primitive.attributes.find(attributeName);

		if (iter == primitive.attributes.end())
		{
			std::cout << attributeName << " : not found !!! \n";
			return;
		}

		const tinygltf::Accessor& accessorAttribute = model.accessors[iter->second];
		const tinygltf::BufferView& bufferView = model.bufferViews[accessorAttribute.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

		if (accessorAttribute.type == TINYGLTF_TYPE_VEC4)
		{
			if (accessorAttribute.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
			{
				const unsigned short* data = reinterpret_cast<const unsigned short*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
				for (size_t i = 0; i < accessorAttribute.count; ++i)
				{

					vec4Data.push_back(Vec4f(float(data[i * 4 + 0]), float(data[i * 4 + 1]), float(data[i * 4 + 2]), float(data[i * 4 + 3])));
				}
			}
			else if (accessorAttribute.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
			{
				const unsigned int* data = reinterpret_cast<const unsigned int*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
				for (size_t i = 0; i < accessorAttribute.count; ++i)
				{
					vec4Data.push_back(Vec4f(float(data[i * 4 + 0]), float(data[i * 4 + 1]), float(data[i * 4 + 2]), float(data[i * 4 + 3])));
				}
			}
			else if (accessorAttribute.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
			{
				const float* data = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
				for (size_t i = 0; i < accessorAttribute.count; ++i)
				{
					vec4Data.push_back(Vec4f(data[i * 4 + 0], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]));
				}
			}
		}


	}

	template<typename TDataType>
	void GltfLoader<TDataType>::getVertexBindJoint(
		tinygltf::Model& model,
		const tinygltf::Primitive& primitive,
		const std::string& attributeName,
		std::vector<Vec4f>& vec4Data,
		const std::vector<int>& skinJoints
	)
	{
		//assign Attributes for Points
		std::map<std::string, int>::const_iterator iter;
		iter = primitive.attributes.find(attributeName);

		if (iter == primitive.attributes.end())
		{
			std::cout << attributeName << " : not found !!! \n";
			return;
		}

		const tinygltf::Accessor& accessorAttribute = model.accessors[iter->second];
		const tinygltf::BufferView& bufferView = model.bufferViews[accessorAttribute.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];


		if (accessorAttribute.type == TINYGLTF_TYPE_VEC4)
		{
			if (accessorAttribute.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
			{
				const unsigned short* data = reinterpret_cast<const unsigned short*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
				for (size_t i = 0; i < accessorAttribute.count; ++i)
				{

					int a = int(data[i * 4 + 0]);
					int b = int(data[i * 4 + 1]);
					int c = int(data[i * 4 + 2]);
					int d = int(data[i * 4 + 3]);

					vec4Data.push_back(Vec4f(skinJoints[a], skinJoints[b], skinJoints[c], skinJoints[d]));

				}
			}
			else if (accessorAttribute.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
			{
				const unsigned int* data = reinterpret_cast<const unsigned int*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
				for (size_t i = 0; i < accessorAttribute.count; ++i)
				{
					int a = int(data[i * 4 + 0]);
					int b = int(data[i * 4 + 1]);
					int c = int(data[i * 4 + 2]);
					int d = int(data[i * 4 + 3]);

					vec4Data.push_back(Vec4f(skinJoints[a], skinJoints[b], skinJoints[c], skinJoints[d]));
				}
			}
			else if (accessorAttribute.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
			{
				const float* data = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
				for (size_t i = 0; i < accessorAttribute.count; ++i)
				{
					int a = int(data[i * 4 + 0]);
					int b = int(data[i * 4 + 1]);
					int c = int(data[i * 4 + 2]);
					int d = int(data[i * 4 + 3]);

					vec4Data.push_back(Vec4f(skinJoints[a], skinJoints[b], skinJoints[c], skinJoints[d]));
				}
			}
		}
	}

	template<typename TDataType>
	void GltfLoader<TDataType>::getJointsTransformData(const std::vector<joint>& all_Joints,
		std::vector<std::string>& joint_name,
		std::vector<std::vector<int>>& joint_child
	)
	{
		for (size_t k = 0; k < all_Joints.size(); k++)
		{
			joint jId = all_Joints[k];
			std::vector<int>& children = model.nodes[jId].children;				//std::vector<int> children ;构造骨骼层级，

			std::vector<double>& rotation = model.nodes[jId].rotation;			//quat length must be 0 or 4
			std::vector<double>& scale = model.nodes[jId].scale;					//length must be 0 or 3
			std::vector<double>& translation = model.nodes[jId].translation;		//length must be 0 or 3
			std::vector<double>& matrix = model.nodes[jId].matrix;				//length must be 0 or 16
			std::vector<double>& MorphTargetWeights = model.nodes[jId].weights;	//The weights of the instantiated Morph Target

			joint_name.push_back(model.nodes[jId].name);
			joint_child.push_back(children);


			if (!rotation.empty())
				joint_rotation[jId] = (Quat<float>(rotation[0], rotation[1], rotation[2], rotation[3]));
			else
				joint_rotation[jId] = (Quat<float>(0, 0, 0, 0));

			if (!scale.empty())
				joint_scale[jId] = (Vec3f(scale[0], scale[1], scale[2]));
			else
				joint_scale[jId] = (Vec3f(1.0f, 1.0f, 1.0f));

			if (!translation.empty())
				joint_translation[jId] = (Vec3f(translation[0], translation[1], translation[2]));
			else
				joint_translation[jId] = (Vec3f(0.0f, 0.0f, 0.0f));

			if (!matrix.empty())
			{
				joint_matrix[jId] = (Mat4f(matrix[0], matrix[4], matrix[8], matrix[12],
					matrix[1], matrix[5], matrix[9], matrix[13],
					matrix[2], matrix[6], matrix[10], matrix[14],
					matrix[3], matrix[7], matrix[11], matrix[15]));

			}
			else
			{
				//Translation Matrix
				Mat4f T;
				if (!translation.empty())
					T = Mat4f(1, 0, 0, translation[0], 0, 1, 0, translation[1], 0, 0, 1, translation[2], 0, 0, 0, 1);
				else
					T = Mat4f::identityMatrix();


				Mat4f R;
				if (!rotation.empty())
					R = Quat<float>(rotation[0], rotation[1], rotation[2], rotation[3]).toMatrix4x4();
				else
					R = Mat4f::identityMatrix();

				Mat4f S;
				if (!scale.empty())
					S = Mat4f(scale[0], 0, 0, 0, 0, scale[1], 0, 0, 0, 0, scale[2], 0, 0, 0, 0, 1);
				else
					S = Mat4f::identityMatrix();

				joint_matrix[jId] = (T * R * S);// if jointmatrix not found, build it

			}
			Quat<float> temp;
			if (!rotation.empty())
				temp = Quat<float>(rotation[0], rotation[1], rotation[2], rotation[3]);
			else
				temp = Quat<float>(-123, -123, -123, -123);
		}
	}




	// ********************************** getVec3f By Attribute Name *************************//
	template<typename TDataType>
	void GltfLoader<TDataType>::getCoordByAttributeName(
		tinygltf::Model& model,
		const tinygltf::Primitive& primitive,
		const std::string& attributeName,
		std::vector<Coord>& vertices
	)
	{
		//assign Attributes for Points
		std::map<std::string, int>::const_iterator iter;
		iter = primitive.attributes.find(attributeName);

		if (iter == primitive.attributes.end())
		{
			std::cout << attributeName << " : not found !!! \n";
			return;
		}

		const tinygltf::Accessor& accessorAttribute = model.accessors[iter->second];
		const tinygltf::BufferView& bufferView = model.bufferViews[accessorAttribute.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

		if (accessorAttribute.type == TINYGLTF_TYPE_VEC3)
		{
			const float* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
			for (size_t i = 0; i < accessorAttribute.count; ++i)
			{
				vertices.push_back(Coord(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]));
			}
		}
		else if (accessorAttribute.type == TINYGLTF_TYPE_VEC2)
		{
			const float* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
			for (size_t i = 0; i < accessorAttribute.count; ++i)
			{
				vertices.push_back(Coord(positions[i * 2 + 0], positions[i * 2 + 1], 0));
			}
		}
	}



	template<typename TDataType>
	void GltfLoader<TDataType>::getRealByIndex(tinygltf::Model& model, int index, std::vector<Real>& result)
	{

		// 获取指定索引处的实数值
		const tinygltf::Accessor& accessor = model.accessors[index];
		const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];


		// 假设实数值是浮点数类型
		if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
		{
			const float* dataPtr = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

			// 将实数值存储在vector中
			for (size_t i = 0; i < accessor.count; ++i) {
				result.push_back(static_cast<Real>(dataPtr[i]));
			}
		}
		else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
		{
			const double* dataPtr = reinterpret_cast<const double*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

			// 将实数值存储在vector中
			for (size_t i = 0; i < accessor.count; ++i) {
				result.push_back(static_cast<Real>(dataPtr[i]));
			}
		}
		else
		{
			printf("\n !!!!!!!!  Error ComponentType  !!!!!!!!\n");
		}

		return;
	}

	template<typename TDataType>
	void GltfLoader<TDataType>::getVec3fByIndex(tinygltf::Model& model, int index, std::vector<Vec3f>& result)
	{
		// 获取指定索引处的实数值
		const tinygltf::Accessor& accessor = model.accessors[index];
		const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];


		if (accessor.type == TINYGLTF_TYPE_VEC3)
		{
			const float* dataPtr = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

			for (size_t i = 0; i < accessor.count; ++i) {
				result.push_back(Vec3f(dataPtr[i * 3 + 0], dataPtr[i * 3 + 1], dataPtr[i * 3 + 2]));
			}
		}


	}

	template<typename TDataType>
	void GltfLoader<TDataType>::getQuatByIndex(tinygltf::Model& model, int index, std::vector<Quat<float>>& result)
	{

		// 获取指定索引处的实数值
		const tinygltf::Accessor& accessor = model.accessors[index];
		const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];


		if (accessor.type == TINYGLTF_TYPE_VEC4)
		{
			const float* dataPtr = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

			// 存储在vector中
			for (size_t i = 0; i < accessor.count; ++i) {
				result.push_back(Quat<float>(dataPtr[i * 4 + 0], dataPtr[i * 4 + 1], dataPtr[i * 4 + 2], dataPtr[i * 4 + 3]));
			}
		}



	}


	template<typename TDataType>
	void GltfLoader<TDataType>::getMatrix(
		tinygltf::Model& model,
		std::vector<Mat4f>& mat
	)
	{
		const tinygltf::Accessor& accessor = model.accessors[model.skins[0].inverseBindMatrices];
		const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];


		if (accessor.type == TINYGLTF_TYPE_MAT4)
		{
			const float* m = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

			for (size_t i = 0; i < accessor.count; ++i)
			{

				mat.push_back(Mat4f(m[i * 16 + 0], m[i * 16 + 1], m[i * 16 + 2], m[i * 16 + 3],
					m[i * 16 + 4], m[i * 16 + 5], m[i * 16 + 6], m[i * 16 + 7],
					m[i * 16 + 8], m[i * 16 + 9], m[i * 16 + 10], m[i * 16 + 11],
					m[i * 16 + 12], m[i * 16 + 13], m[i * 16 + 14], m[i * 16 + 15]
				));

			}
		}
	}


	template<typename TDataType>
	void GltfLoader<TDataType>::updateStates()
	{
		// 更新骨骼点位置
		std::vector<Coord> jointVertices;
		for (size_t j = 0; j < all_Joints.size(); j++)
		{
			joint jId = all_Joints[j];

			if (this->varImportAnimation()->getValue())
			{
				this->updateAnimationMatrix(all_Joints, this->stateFrameNumber()->getValue());
				jointVertices.push_back(getVertexLocationWithJointTransform(jId, Vec3f(0, 0, 0), joint_AnimaMatrix));
			}
			else
			{
				jointVertices.push_back(getVertexLocationWithJointTransform(jId, Vec3f(0, 0, 0), joint_matrix));
			}
		}

		this->stateJointSet()->getDataPtr()->setPoints(jointVertices);

		if (model.animations.size() && this->varImportAnimation()->getValue())
		{
			this->updateAnimationMatrix(all_Joints, this->stateFrameNumber()->getValue());

			this->updateJointWorldMatrix(all_Joints, joint_AnimaMatrix);



			cuExecute(this->stateVertex()->getData().size(),
				PointsAnimation,
				initialPosition,
				this->stateVertex()->getData(),
				this->stateJointInverseBindMatrix()->getData(),
				this->stateJointWorldMatrix()->getData(),

				this->stateBindJoints_0()->getData(),
				this->stateBindJoints_1()->getData(),
				this->stateWeights_0()->getData(),
				this->stateWeights_1()->getData()

			);
		}
	};

	template<typename TDataType>
	std::vector<int> GltfLoader<TDataType>::getJointDirByJointIndex(std::map<int, std::vector<int>> jointId_joint_Dir, int Index)
	{
		std::vector<int> jointDir;
		std::map<int, std::vector<int>>::const_iterator iter;

		//get skeletal chain
		iter = jointId_joint_Dir.find(Index);
		if (iter == jointId_joint_Dir.end())
		{
			std::cout << "Error: not found JointIndex \n";
			return jointDir;
		}

		jointDir = iter->second;		// std::vector<int> jD 按顺序为    
		return jointDir;				//当前骨骼->Root  如 4-2-1-0表示当前骨骼为4，向上依次为 4-2-1-0
	}


	template<typename TDataType>
	void GltfLoader<TDataType>::updateAnimationMatrix(const std::vector<joint>& all_Joints, int currentframe)
	{
		joint_AnimaMatrix = joint_matrix;

		for (auto jId : all_Joints)
		{
			const std::vector<int>& jD = getJointDirByJointIndex(jointId_joint_Dir, jId);	//get joint 骨骼链

			Mat4f tempMatrix = Mat4f::identityMatrix();

			//按骨骼id应用矩阵
			for (int k = jD.size() - 1; k >= 0; k--)
			{

				joint select = jD[k];

				Vec3f vT = Vec3f(0, 0, 0);
				Vec3f vS = Vec3f(1, 1, 1);
				Quat<float> qR = Quat<float>(Mat3f::identityMatrix());

				//replace 
				//	如果有动画,那就用动画的变换，是否有动画可以判断  joint_R_f_anim这三个是否有值。
				//   但是最好判断model.nodes[k].matrix是否为空
				//应用动画，需要单独摘出来
				if (model.nodes[select].matrix.empty())		//只有矩阵为空，才是有动画
				{
					auto iterR = joint_R_f_anim.find(select);
					//auto iterR_inPut = joint_input.find(select);		//这里之后要改成时间、添加关键帧插值
					//是否有动画，如果有动画就用动画的变换

					int tempFrame = 0;
					//防止越界
					if (currentframe > iterR->second.size() - 1)
						tempFrame = iterR->second.size() - 1;
					else if (currentframe < 0)
						tempFrame = 0;
					else
						tempFrame = currentframe;


					if (iterR != joint_R_f_anim.end())
					{
						//使用当前帧
						qR = iterR->second[tempFrame];

					}
					else
					{
						//如果没动画，也没矩阵，那应该有初始变换
						//或者用初始值，已经初始化 qR
						qR = joint_rotation[select];
					}

					std::map<joint, std::vector<Vec3f>>::const_iterator iterT;
					iterT = joint_T_f_anim.find(select);
					if (iterT != joint_T_f_anim.end())
					{
						vT = iterT->second[tempFrame];
					}
					else
					{
						vT = joint_translation[select];
					}

					std::map<joint, std::vector<Vec3f>>::const_iterator iterS;
					iterS = joint_S_f_anim.find(select);
					if (iterS != joint_S_f_anim.end())
					{
						vS = iterS->second[tempFrame];
					}
					else
					{
						vS = joint_scale[select];
					}

					Mat4f mT = Mat4f(1, 0, 0, vT[0], 0, 1, 0, vT[1], 0, 0, 1, vT[2], 0, 0, 0, 1);
					Mat4f mS = Mat4f(vS[0], 0, 0, 0, 0, vS[1], 0, 0, 0, 0, vS[2], 0, 0, 0, 0, 1);
					Mat4f mR = qR.toMatrix4x4();

					joint_AnimaMatrix[select] = mT * mS * mR;	//截至到这里全是动画矩阵应用
				}
				//else 使用原始矩阵，已经被copy
			}
		}
	}

	template<typename TDataType>
	Vec3f GltfLoader<TDataType>::getVertexLocationWithJointTransform(joint jointId, Vec3f inPoint, std::map<joint, Mat4f> jMatrix)
	{

		Vec3f result = Vec3f(0);

		const std::vector<int>& jD = getJointDirByJointIndex(jointId_joint_Dir, jointId);	//get joint 骨骼链

		Mat4f tempMatrix = Mat4f::identityMatrix();

		//按骨骼id应用矩阵
		for (int k = jD.size() - 1; k >= 0; k--)
		{
			joint select = jD[k];
			tempMatrix *= jMatrix[select];		//jD[k] 从root(id为0)向当前骨骼逐个应用变换

		}

		Vec4f jointLocation = tempMatrix * Vec4f(inPoint[0], inPoint[1], inPoint[2], 1);
		result = Coord(jointLocation[0], jointLocation[1], jointLocation[2]);

		return result;
	};

	template<typename TDataType>
	void GltfLoader<TDataType>::updateJointWorldMatrix(const std::vector<joint>& allJoints, std::map<joint, Mat4f> jMatrix)
	{
		std::vector<Mat4f> c_joint_Mat4f;

		this->stateJointWorldMatrix()->resize(allJoints.size());
		c_joint_Mat4f.resize(allJoints.size());

		for (size_t i = 0; i < allJoints.size(); i++)
		{
			joint jointId = allJoints[i];
			const std::vector<int>& jD = getJointDirByJointIndex(jointId_joint_Dir, jointId);	//get joint 骨骼链



			Mat4f tempMatrix = Mat4f::identityMatrix();

			//按骨骼id应用矩阵
			for (int k = jD.size() - 1; k >= 0; k--)
			{
				joint select = jD[k];
				tempMatrix *= jMatrix[select];		//jD[k] 从root(id为0)向当前骨骼逐个应用变换
			}
			c_joint_Mat4f[jointId] = tempMatrix;
		}

		this->stateJointWorldMatrix()->assign(c_joint_Mat4f);

	};

	template<typename TDataType>
	void GltfLoader<TDataType>::buildInverseBindMatrices(const std::vector<joint>& all_Joints)
	{

		std::map<joint, Mat4f> tempJointMatrix = joint_matrix;
		std::vector<Mat4f> temp;

		temp.resize(all_Joints.size());

		for (size_t i = 0; i < all_Joints.size(); i++)
		{
			joint jointId = all_Joints[i];

			const std::vector<int>& jD = getJointDirByJointIndex(jointId_joint_Dir, jointId);

			Mat4f tempMatrix = Mat4f::identityMatrix();


			for (int k = 0; k < jD.size(); k++)
			{
				joint select = jD[k];

				Vec3f vT = Vec3f(0, 0, 0);
				Vec3f vS = Vec3f(1, 1, 1);
				Quat<float> qR = Quat<float>(Mat3f::identityMatrix());


				if (model.nodes[select].matrix.empty())//int k = jD.size() - 1; k >= 0; k--
				{

					qR = joint_rotation[select];

					vT = joint_translation[select];

					vS = joint_scale[select];

					Mat4f mT = Mat4f(1, 0, 0, vT[0], 0, 1, 0, vT[1], 0, 0, 1, vT[2], 0, 0, 0, 1);
					Mat4f mS = Mat4f(vS[0], 0, 0, 0, 0, vS[1], 0, 0, 0, 0, vS[2], 0, 0, 0, 0, 1);
					Mat4f mR = qR.toMatrix4x4();
					//

					tempJointMatrix[select] = mT * mS * mR;
				}

				tempMatrix *= tempJointMatrix[select].inverse();		//jD[k] 从root(id为0)向当前骨骼逐个应用变换

			}

			joint_inverseBindMatrix[jointId] = (tempMatrix);



			temp[jointId] = tempMatrix;

		}

		this->stateJointInverseBindMatrix()->assign(temp);

	};


	template<typename TDataType>
	void GltfLoader<TDataType>::getJointAndHierarchy(std::map<scene, std::vector<int>> Scene_JointsNodesId, std::vector<joint>& all_Joints)
	{
		for (auto it : Scene_JointsNodesId)
		{
			scene sceneId = it.first;
			std::vector<joint> sceneJointRoots = it.second;
			std::map<joint, int> root_jointNum;

			for (size_t n = 0; n < sceneJointRoots.size(); n++)
			{
				int rootNodeId = sceneJointRoots[n];

				std::vector<int> nullvec;
				traverseNode(rootNodeId, all_Joints, jointId_joint_Dir, nullvec);
			}

		}
	}

	template<typename TDataType>
	Vec3f GltfLoader<TDataType>::getmeshPointDeformByJoint(joint jointId, Coord worldPosition, std::map<joint, Mat4f> jMatrix)
	{
		Vec3f offest = worldPosition;

		Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], 1);//

		auto v_world_space = getVertexLocationWithJointTransform(jointId, Vec3f(v_bone_space[0], v_bone_space[1], v_bone_space[2]), jMatrix);

		return v_world_space;
	}


	template< typename Coord, typename Vec4f, typename Mat4f>
	__global__ void PointsAnimation(
		DArray<Coord> intialPosition,
		DArray<Coord> worldPosition,
		DArray<Mat4f> joint_inverseBindMatrix,
		DArray<Mat4f> WorldMatrix,

		DArray<Vec4f> bind_joints_0,
		DArray<Vec4f> bind_joints_1,
		DArray<Vec4f> weights_0,
		DArray<Vec4f> weights_1

	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= worldPosition.size()) return;


		Vec4f result = Vec4f(0, 0, 0, 1);
		Coord offest;

		bool j0 = bind_joints_0.size();
		bool j1 = bind_joints_1.size();

		if (j0)
		{
			for (unsigned int i = 0; i < 4; i++)
			{
				int jointId = int(bind_joints_0[pId][i]);
				Real weight = weights_0[pId][i];

				offest = intialPosition[pId];
				Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], 1);//

				result += (WorldMatrix[jointId] * v_bone_space) * weight;
			}
		}
		if (j1)
		{
			for (unsigned int i = 0; i < 4; i++)
			{
				int jointId = int(bind_joints_1[pId][i]);
				Real weight = weights_1[pId][i];

				offest = intialPosition[pId];
				Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], 1);//

				result += (WorldMatrix[jointId] * v_bone_space) * weight;
			}
		}

		if (j0 | j1)
		{
			worldPosition[pId][0] = result[0];
			worldPosition[pId][1] = result[1];
			worldPosition[pId][2] = result[2];
		}


	}




	template<typename TDataType>
	void GltfLoader<TDataType>::InitializationData()
	{
		joint_rotation.clear();
		joint_scale.clear();
		joint_translation.clear();
		joint_matrix.clear();
		jointId_joint_Dir.clear();
		joint_T_f_anim.clear();
		joint_R_f_anim.clear();
		joint_S_f_anim.clear();
		joint_T_Time.clear();
		joint_S_Time.clear();
		joint_R_Time.clear();
		Scene_Name.clear();
		joint_output.clear();
		joint_input.clear();
		joint_inverseBindMatrix.clear();
		joint_AnimaMatrix.clear();
		Scene_Name.clear();
		all_Joints.clear();
		meshVertex_bind_joint_0.clear();
		meshVertex_bind_joint_1.clear();
		meshVertex_joint_weight_0.clear();
		meshVertex_joint_weight_1.clear();



		this->stateBindJoints_0()->clear();
		this->stateBindJoints_1()->clear();
		this->stateWeights_0()->clear();
		this->stateWeights_1()->clear();
		this->stateCoordChannel_1()->clear();
		this->stateCoordChannel_2()->clear();
		this->stateInitialMatrix()->clear();
		this->stateIntChannel_1()->clear();
		this->stateJointInverseBindMatrix()->clear();
		this->stateJointLocalMatrix()->clear();

		this->stateJointWorldMatrix()->clear();
		this->stateMaterials()->clear();
		this->stateNormal()->clear();
		this->stateRealChannel_1()->clear();
		this->stateShapes()->clear();
		this->stateTexCoord_0()->clear();
		this->stateTexCoord_1()->clear();

		this->stateVertex()->clear();
		this->stateWeights_0()->clear();
		this->stateWeights_1()->clear();
	}

	template<typename TDataType>
	void GltfLoader<TDataType>::importAnimation()
	{
		using namespace tinygltf;
		//input output
		for (size_t i = 0; i < model.nodes.size(); i++)
		{
			joint_output[i] = Vec3i(-1, -1, -1);		//
			joint_input[i] = Vec3f(NULL_TIME, NULL_TIME, NULL_TIME);
		}

		//Reset loading animation  ;     在reset时候载入所有动画数据
		for (size_t i = 0; i < model.animations.size(); i++)
		{
			std::string& name = model.animations[i].name;
			std::vector<AnimationChannel>& channels = model.animations[i].channels;
			std::vector<AnimationSampler>& samplers = model.animations[i].samplers;

			for (size_t j = 0; j < channels.size(); j++)	//channels 每个动画通道导入
			{
				//get sampler info
				int& samplerId = channels[j].sampler;              // required
				joint& joint_nodeId = channels[j].target_node;          // required (index of the node to target) 
				std::string& target_path = channels[j].target_path;  // required in ["translation", "rotation", "scale","weights"]

				//get
				int& input = samplers[samplerId].input;			//real time 
				int& output = samplers[samplerId].output;		//transform bufferid
				std::string& interpolation = samplers[samplerId].interpolation;

				//struct AnimationSampler {
				//	int input;                  // Time
				//	int output;                 // Data
				//	std::string interpolation;  // "LINEAR", "STEP","CUBICSPLINE" or user defined  // string. default "LINEAR"
				//}								

				{
					
					if (target_path == "translation")
					{
						joint_output[joint_nodeId][0] = output;
						joint_input[joint_nodeId][0] = input;
					}
					else if (target_path == "scale")
					{
						joint_output[joint_nodeId][1] = output;
						joint_input[joint_nodeId][0] = input;
					}
					else if (target_path == "rotation")
					{
						joint_output[joint_nodeId][2] = output;
						joint_input[joint_nodeId][0] = input;
					}
				}

				//Reset时导入所有动画
				{
					//out animation data
					std::vector<Vec3f> frame_T_anim;
					std::vector<Quat<float>> frame_R_anim;
					std::vector<Vec3f> frame_S_anim;
					//
					std::vector<Real> frame_T_Time;
					std::vector<Real> frame_R_Time;
					std::vector<Real> frame_S_Time;

					//获取关节的变换 output 数据
					if (target_path == "translation")
					{
						getVec3fByIndex(model, output, frame_T_anim);
						joint_T_f_anim[joint_nodeId] = frame_T_anim;

						getRealByIndex(model, input, frame_T_Time);
						joint_T_Time[joint_nodeId] = frame_T_Time;	//获取关节的位移时间戳 input data
					}
					else if (target_path == "scale")
					{
						getVec3fByIndex(model, output, frame_S_anim);
						joint_S_f_anim[joint_nodeId] = frame_S_anim;
						getRealByIndex(model, input, frame_S_Time);
						joint_S_Time[joint_nodeId] = frame_S_Time;	//获取关节的位移时间戳 input data
					}
					else if (target_path == "rotation")
					{
						getQuatByIndex(model, output, frame_R_anim);
						joint_R_f_anim[joint_nodeId] = frame_R_anim;
						getRealByIndex(model, input, frame_R_Time);
						joint_R_Time[joint_nodeId] = frame_R_Time;	//获取关节的旋转时间戳 input data
					}
				}
			}
		}
	}

	template<typename TDataType>
	GltfLoader<TDataType>::~GltfLoader()
	{
		InitializationData();
		initialPosition.clear();
	}


	DEFINE_CLASS(GltfLoader);
}