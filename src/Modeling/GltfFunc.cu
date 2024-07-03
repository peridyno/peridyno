
#include "GltfFunc.h"



namespace dyno
{

	void getBoundingBoxByName(
		tinygltf::Model& model,
		const tinygltf::Primitive& primitive,
		const std::string& attributeName,
		TAlignedBox3D<Real>& bound,
		Transform3f& transform
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

		auto min = model.accessors[iter->second].minValues;
		auto max = model.accessors[iter->second].maxValues;
		if (min.size() != 3)
		{
			std::cout << attributeName << " : not Vec3f !!! \n";
			return;
		}

		Vec3f v0 = Vec3f(min[0], min[1], min[2]);
		Vec3f v1 = Vec3f(max[0], max[1], max[2]);

		bound = TAlignedBox3D<Real>(v0, v1);

		Vec3f center = (v0 + v1) / 2;
		transform = Transform3f(Vec3f(center), Mat3f::identityMatrix(), Vec3f(1));

	}


	void getVec4ByAttributeName(tinygltf::Model& model,
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


	// ********************************** getVec3f By Attribute Name *************************//

	void getVec3fByAttributeName(
		tinygltf::Model& model,
		const tinygltf::Primitive& primitive,
		const std::string& attributeName,
		std::vector<Vec3f>& vertices
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
				vertices.push_back(Vec3f(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]));
			}
		}
		else if (accessorAttribute.type == TINYGLTF_TYPE_VEC2)
		{
			const float* positions = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
			for (size_t i = 0; i < accessorAttribute.count; ++i)
			{
				vertices.push_back(Vec3f(positions[i * 2 + 0], positions[i * 2 + 1], 0));
			}
		}
	}


	// ***************************** get triangle vertexID *************************** //

	void getTriangles(
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



	void getRealByIndex(tinygltf::Model& model, int index, std::vector<Real>& result)
	{

		const tinygltf::Accessor& accessor = model.accessors[index];
		const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];


		if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
		{
			const float* dataPtr = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

			for (size_t i = 0; i < accessor.count; ++i) {
				result.push_back(static_cast<Real>(dataPtr[i]));
			}
		}
		else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE)
		{
			const double* dataPtr = reinterpret_cast<const double*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

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


	void getVec3fByIndex(tinygltf::Model& model, int index, std::vector<Vec3f>& result)
	{
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


	void getQuatByIndex(tinygltf::Model& model, int index, std::vector<Quat<float>>& result)
	{
		const tinygltf::Accessor& accessor = model.accessors[index];
		const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
		const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

		if (accessor.type == TINYGLTF_TYPE_VEC4)
		{
			const float* dataPtr = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

			for (size_t i = 0; i < accessor.count; ++i) {
				result.push_back(Quat<float>(dataPtr[i * 4 + 0], dataPtr[i * 4 + 1], dataPtr[i * 4 + 2], dataPtr[i * 4 + 3]));
			}
		}



	}



	void getVertexBindJoint(
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

					vec4Data.push_back(Vec4f(skinJoints[int(data[i * 4 + 0])], skinJoints[int(data[i * 4 + 1])], skinJoints[int(data[i * 4 + 2])], skinJoints[int(data[i * 4 + 3])]));

				}
			}
			else if (accessorAttribute.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
			{
				const unsigned int* data = reinterpret_cast<const unsigned int*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
				for (size_t i = 0; i < accessorAttribute.count; ++i)
				{

					vec4Data.push_back(Vec4f(skinJoints[int(data[i * 4 + 0])], skinJoints[int(data[i * 4 + 1])], skinJoints[int(data[i * 4 + 2])], skinJoints[int(data[i * 4 + 3])]));
				}
			}
			else if (accessorAttribute.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
			{
				const float* data = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessorAttribute.byteOffset]);
				for (size_t i = 0; i < accessorAttribute.count; ++i)
				{

					vec4Data.push_back(Vec4f(skinJoints[int(data[i * 4 + 0])], skinJoints[int(data[i * 4 + 1])], skinJoints[int(data[i * 4 + 2])], skinJoints[int(data[i * 4 + 3])]));
				}
			}
		}
	}



	std::string getTexUri(const std::vector<tinygltf::Texture>& textures, const std::vector<tinygltf::Image>& images, int index)
	{
		std::string uri;

		if (index == -1)
			return uri;

		auto& TexSource = textures[index].source;
		auto& TexSampler = textures[index].sampler;

		uri = images[TexSource].uri;

		return uri;
	}


	void getNodesAndHierarchy(tinygltf::Model& model, std::map<scene, std::vector<int>> Scene_JointsNodesId, std::vector<joint>& all_Nodes, std::map<joint, std::vector<int>>& id_Dir)
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
				traverseNode(model, rootNodeId, all_Nodes, id_Dir, nullvec);
			}

		}
	}

	void traverseNode(tinygltf::Model& model, joint id, std::vector<joint>& joint_nodes, std::map<joint, std::vector<int>>& dir, std::vector<joint> currentDir)
	{
		const tinygltf::Node& node = model.nodes[id];
		currentDir.push_back(id);
		joint_nodes.push_back(id);

		for (int childIndex : node.children) {
			const tinygltf::Node& childNode = model.nodes[childIndex];
			traverseNode(model, childIndex, joint_nodes, dir, currentDir);
		}

		std::reverse(currentDir.begin(), currentDir.end());
		dir[id] = currentDir;
	}



	void getJointsTransformData(
		const std::vector<int>& all_Joints,
		std::map<int, std::string>& joint_name,
		std::vector<std::vector<int>>& joint_child,
		std::map<int, Quat<float>>& joint_rotation,
		std::map<int, Vec3f>& joint_scale,
		std::map<int, Vec3f>& joint_translation,
		std::map<int, Mat4f>& joint_matrix,
		tinygltf::Model model
	)
	{
		for (size_t k = 0; k < all_Joints.size(); k++)
		{
			joint jId = all_Joints[k];
			std::vector<int>& children = model.nodes[jId].children;				//std::vector<int> children ;

			std::vector<double>& rotation = model.nodes[jId].rotation;			//quat length must be 0 or 4
			std::vector<double>& scale = model.nodes[jId].scale;					//length must be 0 or 3
			std::vector<double>& translation = model.nodes[jId].translation;		//length must be 0 or 3
			std::vector<double>& matrix = model.nodes[jId].matrix;				//length must be 0 or 16

			joint_name[jId] = model.nodes[jId].name;
			joint_child.push_back(children);

			Mat4f tempT = Mat4f::identityMatrix();
			Mat4f tempR = Mat4f::identityMatrix();
			Mat4f tempS = Mat4f::identityMatrix();

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

				if (!translation.empty())
					tempT = Mat4f(1, 0, 0, translation[0], 0, 1, 0, translation[1], 0, 0, 1, translation[2], 0, 0, 0, 1);
				else
					tempT = Mat4f::identityMatrix();



				if (!rotation.empty())
					tempR = Quat<float>(rotation[0], rotation[1], rotation[2], rotation[3]).toMatrix4x4();
				else
					tempR = Mat4f::identityMatrix();


				if (!scale.empty())
					tempS = Mat4f(scale[0], 0, 0, 0, 0, scale[1], 0, 0, 0, 0, scale[2], 0, 0, 0, 0, 1);
				else
					tempS = Mat4f::identityMatrix();

				joint_matrix[jId] = (tempT * tempR * tempS);// if jointmatrix not found, build it

			}

		}
	}

	std::vector<int> getJointDirByJointIndex(int Index, std::map<joint, std::vector<int>> jointId_joint_Dir)
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

		jointDir = iter->second;
		return jointDir;
	}



	


	void buildInverseBindMatrices(
		const std::vector<joint>& all_Joints,
		std::map<joint, Mat4f>& joint_matrix, int& maxJointId,
		tinygltf::Model& model,
		std::map<joint, Quat<float>>& joint_rotation,
		std::map<joint, Vec3f>& joint_translation,
		std::map<joint, Vec3f>& joint_scale,
		std::map<joint, Mat4f>& joint_inverseBindMatrix,
		std::map<joint, std::vector<int>> jointId_joint_Dir

	)
	{

		std::map<joint, Mat4f> tempJointMatrix = joint_matrix;
		std::vector<Mat4f> temp;

		temp.resize(maxJointId + 1);

		for (size_t i = 0; i < maxJointId + 1; i++)
		{
			temp.push_back(Mat4f::identityMatrix());
		}



		for (size_t i = 0; i < all_Joints.size(); i++)
		{
			joint jointId = all_Joints[i];

			const std::vector<int>& jD = getJointDirByJointIndex(jointId,jointId_joint_Dir);

			Mat4f tempMatrix = Mat4f::identityMatrix();


			for (int k = 0; k < jD.size(); k++)
			{
				joint select = jD[k];

				Vec3f tempVT = Vec3f(0, 0, 0);
				Vec3f tempVS = Vec3f(1, 1, 1);
				Quat<float> tempQR = Quat<float>(Mat3f::identityMatrix());

				if (model.nodes[select].matrix.empty())
				{
					tempQR = joint_rotation[select];

					tempVT = joint_translation[select];

					tempVS = joint_scale[select];

					Mat4f mT = Mat4f(1, 0, 0, tempVT[0], 0, 1, 0, tempVT[1], 0, 0, 1, tempVT[2], 0, 0, 0, 1);
					Mat4f mS = Mat4f(tempVS[0], 0, 0, 0, 0, tempVS[1], 0, 0, 0, 0, tempVS[2], 0, 0, 0, 0, 1);
					Mat4f mR = tempQR.toMatrix4x4();
					//

					tempJointMatrix[select] = mT * mS * mR;
				}

				tempMatrix *= tempJointMatrix[select].inverse();

			}

			joint_inverseBindMatrix[jointId] = (tempMatrix);



			temp[jointId] = tempMatrix;

		}

	//	this->stateJointInverseBindMatrix()->assign(temp); //!!!

	};


	void updateJoint_Mesh_Camera_Dir(
		tinygltf::Model& model,
		int& jointNum,
		int& meshNum,
		std::map<joint, std::vector<int>>& jointId_joint_Dir,
		std::vector<joint>& all_Joints,
		std::vector<int>& all_Nodes,
		std::map<joint, std::vector<int>> nodeId_Dir,
		std::map<int, std::vector<int>>& meshId_Dir,
		std::vector<int>& all_Meshs,
		DArray<int>& d_joints,
		int& maxJointId
	)
	{
		for (auto nId : all_Nodes)
		{
			if (model.nodes[nId].mesh == -1 && model.nodes[nId].camera == -1)
			{
				all_Joints.push_back(nId);
				jointId_joint_Dir[nId] = nodeId_Dir[nId];
			}
			if (model.nodes[nId].mesh == 1)
			{
				meshId_Dir[nId] = nodeId_Dir[nId];
			}

		}

		jointNum = all_Joints.size();
		meshNum = all_Meshs.size();

		d_joints.assign(all_Joints);

		if (all_Joints.size())
			maxJointId = *std::max_element(all_Joints.begin(), all_Joints.end());
		else
			maxJointId = -1;
	}


	void getMeshMatrix(
		tinygltf::Model& model,
		const std::vector<int>& all_MeshNodeIDs,
		int& maxMeshId,
		CArray<Mat4f>& mesh_Matrix
	)
	{
		maxMeshId = *std::max_element(all_MeshNodeIDs.begin(), all_MeshNodeIDs.end());

		mesh_Matrix.resize(maxMeshId + 1);

		Mat4f tempT;
		Mat4f tempR;
		Mat4f tempS;

		for (size_t k = 0; k < all_MeshNodeIDs.size(); k++)
		{
			std::vector<double>& rotation = model.nodes[all_MeshNodeIDs[k]].rotation;			//quat length must be 0 or 4
			std::vector<double>& scale = model.nodes[all_MeshNodeIDs[k]].scale;					//length must be 0 or 3
			std::vector<double>& translation = model.nodes[all_MeshNodeIDs[k]].translation;		//length must be 0 or 3
			std::vector<double>& matrix = model.nodes[all_MeshNodeIDs[k]].matrix;				//length must be 0 or 16

			if (!matrix.empty())
			{
				mesh_Matrix[all_MeshNodeIDs[k]] = (Mat4f(matrix[0], matrix[4], matrix[8], matrix[12],
					matrix[1], matrix[5], matrix[9], matrix[13],
					matrix[2], matrix[6], matrix[10], matrix[14],
					matrix[3], matrix[7], matrix[11], matrix[15]));
			}

			else
			{
				//Translation Matrix

				if (!translation.empty())
					tempT = Mat4f(1, 0, 0, translation[0], 0, 1, 0, translation[1], 0, 0, 1, translation[2], 0, 0, 0, 1);
				else
					tempT = Mat4f::identityMatrix();


				if (!rotation.empty())
					tempR = Quat<float>(rotation[0], rotation[1], rotation[2], rotation[3]).toMatrix4x4();
				else
					tempR = Mat4f::identityMatrix();


				if (!scale.empty())
					tempS = Mat4f(scale[0], 0, 0, 0, 0, scale[1], 0, 0, 0, 0, scale[2], 0, 0, 0, 0, 1);
				else
					tempS = Mat4f::identityMatrix();

				mesh_Matrix[all_MeshNodeIDs[k]] = (tempT * tempR * tempS);// if jointmatrix not found, build it
			}
		}

	}




	void importAnimation(
		tinygltf::Model model,
		std::map<joint, Vec3i>& joint_output,
		std::map<joint, Vec3f>& joint_input,
		std::map<joint, std::vector<Vec3f>>& joint_T_f_anim,
		std::map<joint, std::vector<Real>>& joint_T_Time,
		std::map<joint, std::vector<Vec3f>>& joint_S_f_anim,
		std::map<joint, std::vector<Real>>& joint_S_Time,
		std::map<joint, std::vector<Quat<float>>>& joint_R_f_anim,
		std::map<joint, std::vector<Real>>& joint_R_Time
	)
	{
		using namespace tinygltf;
		//input output
		for (size_t i = 0; i < model.nodes.size(); i++)
		{
			joint_output[i] = Vec3i(-1, -1, -1);		//
			joint_input[i] = Vec3f(NULL_TIME, NULL_TIME, NULL_TIME);
		}

		//Reset loading animation  ;   
		for (size_t i = 0; i < model.animations.size(); i++)
		{
			std::string& name = model.animations[i].name;
			std::vector<AnimationChannel>& channels = model.animations[i].channels;
			std::vector<AnimationSampler>& samplers = model.animations[i].samplers;

			for (size_t j = 0; j < channels.size(); j++)	//channels 
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

				//Reset
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



	template< typename Coord, typename Vec4f, typename Mat4f, typename Vec2u>
	__global__ void PointsAnimation(
		DArray<Coord> intialPosition,
		DArray<Coord> worldPosition,
		DArray<Mat4f> joint_inverseBindMatrix,
		DArray<Mat4f> WorldMatrix,

		DArray<Vec4f> bind_joints_0,
		DArray<Vec4f> bind_joints_1,
		DArray<Vec4f> weights_0,
		DArray<Vec4f> weights_1,

		Mat4f transform,
		bool isNormal,

		Vec2u range
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= worldPosition.size()) return;

		if (pId<range[0] || pId>range[1])
			return;

		Vec4f result = Vec4f(0, 0, 0, float(!isNormal));

		int skinInfoVertexId = pId - range[0];

		Coord offest;

		bool j0 = bind_joints_0.size();
		bool j1 = bind_joints_1.size();

		if (j0)
		{
			for (unsigned int i = 0; i < 4; i++)
			{
				int jointId = int(bind_joints_0[skinInfoVertexId][i]);
				Real weight = weights_0[skinInfoVertexId][i];

				offest = intialPosition[pId];
				Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], float(!isNormal));//

				result += (transform * WorldMatrix[jointId] * v_bone_space) * weight;
			}
		}
		if (j1)
		{
			for (unsigned int i = 0; i < 4; i++)
			{
				int jointId = int(bind_joints_1[skinInfoVertexId][i]);
				Real weight = weights_1[skinInfoVertexId][i];

				offest = intialPosition[pId];
				Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], float(!isNormal));//

				result += (WorldMatrix[jointId] * v_bone_space) * weight;
			}
		}

		//result = transform * result;

		if (j0 | j1)
		{
			worldPosition[pId][0] = result[0];
			worldPosition[pId][1] = result[1];
			worldPosition[pId][2] = result[2];
		}

		if (isNormal)
			worldPosition[pId] = worldPosition[pId].normalize();

	}

	template< typename Vec3f, typename Vec4f, typename Mat4f, typename Vec2u>
	void skinAnimation(
		DArray<Vec3f>& intialPosition,
		DArray<Vec3f>& worldPosition,
		DArray<Mat4f>& joint_inverseBindMatrix,
		DArray<Mat4f>& WorldMatrix,

		DArray<Vec4f>& bind_joints_0,
		DArray<Vec4f>& bind_joints_1,
		DArray<Vec4f>& weights_0,
		DArray<Vec4f>& weights_1,

		Mat4f transform,
		bool isNormal,

		Vec2u range
	)
	{

		cuExecute(intialPosition.size(),
			PointsAnimation,
			intialPosition,
			worldPosition,
			joint_inverseBindMatrix,
			WorldMatrix,

			bind_joints_0,
			bind_joints_1,
			weights_0,
			weights_1,
			transform,
			isNormal,
			range
		);

	}

	template void skinAnimation<Vec3f, Vec4f , Mat4f, Vec2u>(DArray<Vec3f>& intialPosition,
		DArray<Vec3f>& worldPosition,
		DArray<Mat4f>& joint_inverseBindMatrix,
		DArray<Mat4f>& WorldMatrix,

		DArray<Vec4f>& bind_joints_0,
		DArray<Vec4f>& bind_joints_1,
		DArray<Vec4f>& weights_0,
		DArray<Vec4f>& weights_1,

		Mat4f transform,
		bool isNormal,

		Vec2u range
		);

}
