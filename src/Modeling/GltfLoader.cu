#include "GltfLoader.h"
#include <GLPhotorealisticRender.h>

#define NULL_TIME (-9599.99)

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tinygltf/tiny_gltf.h"

#include "GltfFunc.h"

namespace dyno
{



	template<typename TDataType>
	GltfLoader<TDataType>::GltfLoader()
	{
		auto callback = std::make_shared<FCallBackFunc>(std::bind(&GltfLoader<TDataType>::varChanged, this));

		this->stateJointSet()->setDataPtr(std::make_shared<EdgeSet<DataType3f>>());
		this->stateShapeCenter()->setDataPtr(std::make_shared<PointSet<DataType3f>>());

		this->varImportAnimation()->attach(callback);
		this->varFileName()->attach(callback);

		auto callbackTransform = std::make_shared<FCallBackFunc>(std::bind(&GltfLoader<TDataType>::updateTransform, this));

		this->varLocation()->attach(callbackTransform);
		this->varScale()->attach(callbackTransform);
		this->varRotation()->attach(callbackTransform);

		//this->varLocation()->attach(callback);
		//this->varScale()->attach(callback);
		//this->varRotation()->attach(callback);

		this->stateTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());

		auto render = std::make_shared<GLPhotorealisticRender>();
		this->stateTextureMesh()->connect(render->inTextureMesh());
		this->graphicsPipeline()->pushModule(render);

		//Joint Render
		auto callbackRender = std::make_shared<FCallBackFunc>(std::bind(&GltfLoader<TDataType>::varRenderChanged, this));
		this->varJointRadius()->attach(callbackRender);

		jointPointRender = std::make_shared<GLPointVisualModule>();
		jointPointRender->setColor(Color(1.0f, 0.0f, 0.0f));
		jointPointRender->varPointSize()->setValue(this->varJointRadius()->getValue());
		jointPointRender->setVisible(true);
		this->stateJointSet()->connect(jointPointRender->inPointSet());
		this->graphicsPipeline()->pushModule(jointPointRender);

		jointLineRender = std::make_shared<GLWireframeVisualModule>();
		jointLineRender->varBaseColor()->setValue(Color(0, 1, 0));
		jointLineRender->setVisible(true);
		jointLineRender->varRadius()->setValue(this->varJointRadius()->getValue() / 3);
		jointLineRender->varRenderMode()->setCurrentKey(GLWireframeVisualModule::EEdgeMode::CYLINDER);
		this->stateJointSet()->connect(jointLineRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(jointLineRender);

		auto glShapeCenter = std::make_shared<GLPointVisualModule>();
		glShapeCenter->setColor(Color(1.0f, 1.0f, 0.0f));
		glShapeCenter->varPointSize()->setValue(this->varJointRadius()->getValue() * 8);
		glShapeCenter->setVisible(true);
		this->stateShapeCenter()->connect(glShapeCenter->inPointSet());
		this->graphicsPipeline()->pushModule(glShapeCenter);

		this->stateTextureMesh()->promoteOuput();
	}





	template<typename TDataType>
	void GltfLoader<TDataType>::varChanged()
	{
		if (this->varFileName()->isEmpty())
			return;

		this->updateTransformState();

		printf("!!!!!!!!!!!!!!!!!    Import GLTF   !!!!!!!!!!!!!!!!!!!!!!!!\n\n\n");

		this->InitializationData();


		using namespace tinygltf;

		auto newModel = new Model;
		Model model = *newModel;
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

		////import Animation
		importAnimation(model, joint_output, joint_input, joint_T_f_anim, joint_T_Time, joint_S_f_anim, joint_S_Time, joint_R_f_anim, joint_R_Time);



		// import Scenes:
		std::map<scene, std::vector<int>> Scene_Nodes;
		for (size_t i = 0; i < model.scenes.size(); i++)
		{
			std::vector<int> vecS_Roots;
			vecS_Roots = model.scenes[i].nodes;
			Scene_Nodes[i] = vecS_Roots;
		}

		getNodesAndHierarchy(model, Scene_Nodes, all_Nodes, nodeId_Dir);	//jointId_joint_Dir	//update private: std::map<joint, std::vector<int>> jointId_joint_Dir;

		updateJoint_Mesh_Camera_Dir(model, jointNum, meshNum, jointId_joint_Dir, all_Joints, all_Nodes, nodeId_Dir, meshId_Dir, all_Meshs, d_joints, maxJointId);

		std::vector<std::vector<int>> joint_child;	//build edgeset;

		//get Local Transform T S R M 
		getJointsTransformData(all_Nodes, joint_Name, joint_child, joint_rotation, joint_scale, joint_translation, joint_matrix, model);

		//get InverseBindMatrix (Global)
		this->buildInverseBindMatrices(all_Nodes);


		std::vector<Mat4f> localMatrix;
		localMatrix.resize(all_Joints.size());
		for (size_t k = 0; k < all_Joints.size(); k++)
		{
			localMatrix[k] = joint_matrix[k];
		}
		this->stateJointLocalMatrix()->assign(localMatrix);


		// get joint World Location	
		printf("************  Set Joint  ************\n");
		{
			std::vector<Coord> jointVertices;
			for (size_t j = 0; j < jointNum; j++)
			{
				joint jId = all_Joints[j];

				jointVertices.push_back(getVertexLocationWithJointTransform(jId, Vec3f(0, 0, 0), joint_matrix));

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

			jointVertices.clear();
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

		//materials
		loadMaterial(model, this->stateTextureMesh()->getDataPtr(), this->varFileName()->getValue());


		//shapes
		auto texMesh = this->stateTextureMesh()->getDataPtr();

		auto& reShapes = texMesh->shapes();
		reShapes.clear();
		reShapes.resize(shapeNum);

		auto& reMats = texMesh->materials();



		std::vector<Coord> shapeCenter;

		int primitive_PointOffest;
		int currentShape = 0;


		std::map<int, int> shape_meshId;

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
				getVec3fByAttributeName(model, primitive, std::string("POSITION"), vertices);


				//Set Normal
				getVec3fByAttributeName(model, primitive, std::string("NORMAL"), normals);

				//Set TexCoord

				getVec3fByAttributeName(model, primitive, std::string("TEXCOORD_0"), texCoord0);

				getVec3fByAttributeName(model, primitive, std::string("TEXCOORD_1"), texCoord1);



				//Set Triangles

				if (primitive.mode == TINYGLTF_MODE_TRIANGLES)
				{

					std::vector<TopologyModule::Triangle> tempTriangles;

					getTriangles(model, primitive, tempTriangles, primitive_PointOffest);

					vertexIndex = (tempTriangles);
					normalIndex = (tempTriangles);
					texCoordIndex = (tempTriangles);

					reShapes[currentShape] = std::make_shared<Shape>();
					shape_meshId[currentShape] = mId;		// set shapeId - meshId;
					reShapes[currentShape]->vertexIndex.assign(vertexIndex);
					reShapes[currentShape]->normalIndex.assign(normalIndex);
					reShapes[currentShape]->texCoordIndex.assign(texCoordIndex);

					getBoundingBoxByName(model, primitive, std::string("POSITION"), reShapes[currentShape]->boundingBox, reShapes[currentShape]->boundingTransform);//,Transform3f& transform

					shapeCenter.push_back(reShapes[currentShape]->boundingTransform.translation());

					int matId = primitive.material;
					if (matId != -1 && matId < reMats.size())//
					{
						reShapes[currentShape]->material = reMats[matId];
						//printf("shape_materialID : %d - %d\n",currentShape,matId);

						//printf("texture size %d - %d:\n", reMats[matId]->texColor.nx(), reMats[matId]->texColor.ny());
					}
					else
					{
						reShapes[currentShape]->material = NULL;
					}

					//else //
					//{
					//	auto newMat = std::make_shared<Material>();

					//	newMat->ambient = { 0,0,0 };
					//	newMat->diffuse = Vec3f(0.5, 0.5, 0.5);
					//	newMat->alpha = 1;
					//	newMat->specular = Vec3f(1, 1, 1);
					//	newMat->roughness = 0.5;

					//	reMats.push_back(newMat);

					//	reShapes[currentShape]->material = reMats[reMats.size() - 1];
					//	printf("shape_materialID : %d - %d\n", currentShape, currentShape);
					//}



					trianglesVector.insert(trianglesVector.end(), tempTriangles.begin(), tempTriangles.end());

				}
				currentShape++;

			}
			this->stateShapeCenter()->getDataPtr()->setPoints(shapeCenter);
		}



		std::map<uint, uint> vertexId_shapeId;

		
		texMesh->shapeIds().resize(vertices.size());

		//Scene_SkinNodesId;

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




		for (int i = 0; i < texMesh->shapes().size(); i++)
		{
			auto it = texMesh->shapes()[i];
			//printf(" i = %d, triangle = %d, v = %d \n",i, texMesh->shapes()[i]->vertexIndex.size(),vertex_shapeId.size());

			cuExecute(texMesh->shapes()[i]->vertexIndex.size(),
				updateVertexId_Shape,
				texMesh->shapes()[i]->vertexIndex,
				texMesh->shapeIds(),		//vertex_shapeId, 
				i
			);
		}

		CArray<int> c_shape_meshId;
		DArray<int> d_shape_meshId;
		c_shape_meshId.resize(shape_meshId.size());

		std::vector<int> MeshNodeIDs;

		for (size_t nId = 0; nId < model.nodes.size(); nId++)
		{
			int j = 0;

			if (model.nodes[nId].mesh >= 0)
			{
				j++;
				MeshNodeIDs.push_back(nId);
			}

		}

		getMeshMatrix(model, MeshNodeIDs, maxMeshId, mesh_Matrix);


		for (auto it : shape_meshId)
		{
			c_shape_meshId[it.first] = MeshNodeIDs[it.second];
		}
		d_shape_meshId.assign(c_shape_meshId);




		initialPosition.assign(vertices);
		initialNormal.assign(normals);

		texMesh->vertices().assign(vertices);
		texMesh->normals().assign(normals);



		DArray<Mat4f> d_mesh_Matrix;
		d_mesh_Matrix.assign(mesh_Matrix);

		cuExecute(texMesh->vertices().size(),
			ShapeTransform,
			initialPosition,
			texMesh->vertices(),
			initialNormal,
			texMesh->normals(),
			d_mesh_Matrix,
			texMesh->shapeIds(),
			d_shape_meshId
		);




		texMesh->shapeIds().resize(texMesh->vertices().size());



		// update VertexId_ShapeId
		if (ToCenter)
		{


			d_ShapeCenter.assign(shapeCenter);

			cuExecute(texMesh->vertices().size(),
				ShapeToCenter,
				initialPosition,
				texMesh->vertices(),
				texMesh->shapeIds(),
				d_ShapeCenter,
				this->stateTransform()->getValue()
			);

		}





		std::vector<Vec2f> tempTexCoord;
		for (auto uv0 : texCoord0)
		{
			tempTexCoord.push_back(Vec2f(uv0[0], 1 - uv0[1]));	// uv.v need flip
		}
		this->stateTexCoord_0()->assign(tempTexCoord);
		texMesh->texCoords().assign(tempTexCoord);


		tempTexCoord.clear();
		for (auto uv1 : texCoord1)
		{
			tempTexCoord.push_back(Vec2f(uv1[0], 1 - uv1[1]));
		}
		this->stateTexCoord_1()->assign(tempTexCoord);
		texCoord1.clear();
		tempTexCoord.clear();


		if (all_Joints.empty())
		{
			auto vL = this->varLocation()->getValue();
			auto vS = this->varScale()->getValue();

			Quat<float> q = computeQuaternion();

			auto RV = [&](const Coord& v)->Coord {
				return vL + q.rotate(v - vL);
			};

			int numpt = vertices.size();

			for (int i = 0; i < numpt; i++)
			{
				vertices[i] = RV(vertices[i] * vS + RV(vL));
			}
			texMesh->vertices().assign(vertices);
			initialPosition.assign(vertices);
		}


		this->updateTransform();

	}



	// ***************************** function *************************** //




	template<typename TDataType>
	void GltfLoader<TDataType>::updateStates()
	{
		if (joint_output.empty() || this->varImportAnimation()->getValue())
			return;

		updateAnimation(this->stateFrameNumber()->getValue());
	};


	template<typename TDataType>
	void GltfLoader<TDataType>::updateAnimation(int frameNumber)
	{
		auto mesh = this->stateTextureMesh()->getDataPtr();

		this->updateAnimationMatrix(all_Joints, frameNumber);
		this->updateJointWorldMatrix(all_Joints, joint_AnimaMatrix);



		//update Joints
		cuExecute(all_Joints.size(),
			jointAnimation,
			this->stateJointSet()->getDataPtr()->getPoints(),
			this->stateJointWorldMatrix()->getData(),
			d_joints,
			this->stateTransform()->getValue()
		);

		//update Points
		cuExecute(mesh->vertices().size(),
			PointsAnimation,
			initialPosition,
			mesh->vertices(),
			this->stateJointInverseBindMatrix()->getData(),
			this->stateJointWorldMatrix()->getData(),

			this->stateBindJoints_0()->getData(),
			this->stateBindJoints_1()->getData(),
			this->stateWeights_0()->getData(),
			this->stateWeights_1()->getData(),
			this->stateTransform()->getValue(),
			false
		);

		//update Normals
		cuExecute(mesh->vertices().size(),
			PointsAnimation,
			initialNormal,
			mesh->normals(),
			this->stateJointInverseBindMatrix()->getData(),
			this->stateJointWorldMatrix()->getData(),

			this->stateBindJoints_0()->getData(),
			this->stateBindJoints_1()->getData(),
			this->stateWeights_0()->getData(),
			this->stateWeights_1()->getData(),
			this->stateTransform()->getValue(),
			true
		);

	};


	template<typename TDataType>
	std::vector<int> GltfLoader<TDataType>::getJointDirByJointIndex(int Index)
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



	template<typename TDataType>
	void GltfLoader<TDataType>::updateAnimationMatrix(const std::vector<joint>& all_Joints, int currentframe)
	{
		joint_AnimaMatrix = joint_matrix;

		for (auto jId : all_Joints)
		{
			const std::vector<int>& jD = getJointDirByJointIndex(jId);

			Mat4f tempMatrix = Mat4f::identityMatrix();

			//
			for (int k = jD.size() - 1; k >= 0; k--)
			{

				joint select = jD[k];

				Vec3f tempVT = Vec3f(0, 0, 0);
				Vec3f tempVS = Vec3f(1, 1, 1);
				Quat<float> tempQR = Quat<float>(Mat3f::identityMatrix());

				//replace 

				if (joint_input.find(select) != joint_input.end())		//ֻ
				{
					auto iterR = joint_R_f_anim.find(select);
					//auto iterR_inPut = joint_input.find(select);		//

					int tempFrame = 0;

					if (iterR != joint_R_f_anim.end())
					{

						if (currentframe > iterR->second.size() - 1)
							tempFrame = iterR->second.size() - 1;
						else if (currentframe < 0)
							tempFrame = 0;
						else
							tempFrame = currentframe;

						tempQR = iterR->second[tempFrame];
					}
					else
					{
						tempQR = joint_rotation[select];
					}

					std::map<joint, std::vector<Vec3f>>::const_iterator iterT;
					iterT = joint_T_f_anim.find(select);
					if (iterT != joint_T_f_anim.end())
					{
						//
						if (currentframe > iterT->second.size() - 1)
							tempFrame = iterT->second.size() - 1;
						else if (currentframe < 0)
							tempFrame = 0;
						else
							tempFrame = currentframe;

						tempVT = iterT->second[tempFrame];
					}
					else
					{
						tempVT = joint_translation[select];
					}

					std::map<joint, std::vector<Vec3f>>::const_iterator iterS;
					iterS = joint_S_f_anim.find(select);
					if (iterS != joint_S_f_anim.end())
					{
						//
						if (currentframe > iterS->second.size() - 1)
							tempFrame = iterS->second.size() - 1;
						else if (currentframe < 0)
							tempFrame = 0;
						else
							tempFrame = currentframe;

						tempVS = iterS->second[tempFrame];
					}
					else
					{
						tempVS = joint_scale[select];
					}

					Mat4f mT = Mat4f(1, 0, 0, tempVT[0], 0, 1, 0, tempVT[1], 0, 0, 1, tempVT[2], 0, 0, 0, 1);
					Mat4f mS = Mat4f(tempVS[0], 0, 0, 0, 0, tempVS[1], 0, 0, 0, 0, tempVS[2], 0, 0, 0, 0, 1);
					Mat4f mR = tempQR.toMatrix4x4();

					joint_AnimaMatrix[select] = mT * mS * mR;	//
				}
				//
			}
		}
	}

	template<typename TDataType>
	Vec3f GltfLoader<TDataType>::getVertexLocationWithJointTransform(joint jointId, Vec3f inPoint, std::map<joint, Mat4f> jMatrix)
	{

		Vec3f result = Vec3f(0);

		const std::vector<int>& jD = getJointDirByJointIndex(jointId);

		Mat4f tempMatrix = Mat4f::identityMatrix();

		//
		for (int k = jD.size() - 1; k >= 0; k--)
		{
			joint select = jD[k];
			tempMatrix *= jMatrix[select];		//

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
		c_joint_Mat4f.resize(maxJointId + 1);


		for (size_t i = 0; i < maxJointId + 1; i++)
		{
			c_joint_Mat4f.push_back(Mat4f::identityMatrix());
		}

		for (size_t i = 0; i < allJoints.size(); i++)
		{
			joint jointId = allJoints[i];
			const std::vector<int>& jD = getJointDirByJointIndex(jointId);



			Mat4f tempMatrix = Mat4f::identityMatrix();

			//
			for (int k = jD.size() - 1; k >= 0; k--)
			{
				joint select = jD[k];
				tempMatrix *= jMatrix[select];		//
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

		temp.resize(maxJointId + 1);

		for (size_t i = 0; i < maxJointId + 1; i++)
		{
			temp.push_back(Mat4f::identityMatrix());
		}



		for (size_t i = 0; i < all_Joints.size(); i++)
		{
			joint jointId = all_Joints[i];

			const std::vector<int>& jD = getJointDirByJointIndex(jointId);

			Mat4f tempMatrix = Mat4f::identityMatrix();


			for (int k = 0; k < jD.size(); k++)
			{
				joint select = jD[k];

				Vec3f tempVT = Vec3f(0, 0, 0);
				Vec3f tempVS = Vec3f(1, 1, 1);
				Quat<float> tempQR = Quat<float>(Mat3f::identityMatrix());

				if (joint_input.find(select) != joint_input.end())
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

		this->stateJointInverseBindMatrix()->assign(temp);

	};


	template<typename TDataType>
	Vec3f GltfLoader<TDataType>::getmeshPointDeformByJoint(joint jointId, Coord worldPosition, std::map<joint, Mat4f> jMatrix)
	{
		Vec3f offest = worldPosition;

		Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], 1);//

		auto v_world_space = getVertexLocationWithJointTransform(jointId, Vec3f(v_bone_space[0], v_bone_space[1], v_bone_space[2]), jMatrix);

		return v_world_space;
	}

	template<typename TDataType>
	void GltfLoader<TDataType>::updateTransformState()
	{
		Vec3f location = this->varLocation()->getValue();
		Vec3f scale = this->varScale()->getValue();
		Mat4f mT = Mat4f(1, 0, 0, location[0], 0, 1, 0, location[1], 0, 0, 1, location[2], 0, 0, 0, 1);
		Mat4f mS = Mat4f(scale[0], 0, 0, 0, 0, scale[1], 0, 0, 0, 0, scale[2], 0, 0, 0, 0, 1);
		Mat4f mR = computeQuaternion().toMatrix4x4();
		Mat4f transform = mT * mS * mR;

		this->stateTransform()->setValue(transform);
	}

	template<typename TDataType>
	void GltfLoader<TDataType>::updateTransform()
	{
		this->updateTransformState();

		if (all_Joints.size())
		{
			if (ToCenter)
			{
				cuExecute(this->stateTextureMesh()->getDataPtr()->vertices().size(),
					ShapeToCenter,
					initialPosition,
					this->stateTextureMesh()->getDataPtr()->vertices(),
					this->stateTextureMesh()->getDataPtr()->shapeIds(),
					d_ShapeCenter,
					this->stateTransform()->getValue()
				);
			}
			else
				updateAnimation(this->stateFrameNumber()->getValue());
		}
		else
		{
			if (this->stateTextureMesh()->getDataPtr()->vertices().size())
			{
				//setStaticMeshTransform(initialPosition, this->stateTextureMesh()->getDataPtr()->vertices(), transform);

				cuExecute(this->stateTextureMesh()->getDataPtr()->vertices().size(),
					StaticMeshTransform,
					initialPosition,
					this->stateTextureMesh()->getDataPtr()->vertices(),
					this->stateTransform()->getValue()
				);


			}

		}

	}


	template< typename Coord, typename Mat4f>
	__global__ void StaticMeshTransform(
		DArray<Coord> initialPosition,
		DArray<Coord> Position,
		Mat4f transform
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Position.size()) return;

		Vec4f temp = transform * Vec4f(initialPosition[pId][0], initialPosition[pId][1], initialPosition[pId][2], 1);
		Position[pId] = Vec3f(temp[0], temp[1], temp[2]);

	}


	template< typename Coord, typename Mat4f, typename Vec3f >
	__global__ void ShapeTransform(
		DArray<Coord> intialPosition,
		DArray<Vec3f> worldPosition,
		DArray<Coord> intialNormal,
		DArray<Vec3f> Normal,
		DArray<Mat4f> WorldMatrix,
		DArray<uint> vertexId_shape,
		DArray<int> shapeId_MeshId
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intialPosition.size()) return;

		int shape = vertexId_shape[pId];

		int MeshId = shapeId_MeshId[shape];

		Vec4f tempV = Vec4f(intialPosition[pId][0], intialPosition[pId][1], intialPosition[pId][2], 1);
		Vec4f tempN = Vec4f(intialNormal[pId][0], intialNormal[pId][1], intialNormal[pId][2], 0);

		tempV = WorldMatrix[MeshId] * tempV;
		tempN = WorldMatrix[MeshId] * tempN;

		worldPosition[pId] = Coord(tempV[0], tempV[1], tempV[2]);
		Normal[pId] = Coord(tempN[0], tempN[1], tempN[2]);

	}


	template<typename Triangle>
	__global__ void updateVertexId_Shape(
		DArray<Triangle> triangle,
		DArray<uint> ID_shapeId,
		int shapeId
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangle.size()) return;

		ID_shapeId[triangle[pId][0]] = shapeId;
		ID_shapeId[triangle[pId][1]] = shapeId;
		ID_shapeId[triangle[pId][2]] = shapeId;

	}



	template< typename Coord, typename Vec4f, typename Mat4f >
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
		bool isNormal
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= worldPosition.size()) return;


		Vec4f result = Vec4f(0, 0, 0, float(!isNormal));


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
				Vec4f v_bone_space = joint_inverseBindMatrix[jointId] * Vec4f(offest[0], offest[1], offest[2], float(!isNormal));//

				result += (transform * WorldMatrix[jointId] * v_bone_space) * weight;
			}
		}
		if (j1)
		{
			for (unsigned int i = 0; i < 4; i++)
			{
				int jointId = int(bind_joints_1[pId][i]);
				Real weight = weights_1[pId][i];

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

	template< typename Coord, typename Mat4f>
	__global__ void jointAnimation(
		DArray<Coord> worldPosition,
		DArray<Mat4f> WorldMatrix,
		DArray<int> joints,
		Mat4f transform
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= joints.size()) return;

		Vec4f result = Vec4f(0, 0, 0, 1);
		Coord offest;
		int jointId = joints[pId];
		result = transform * WorldMatrix[jointId] * result;

		//result = transform * result;

		worldPosition[pId][0] = result[0];
		worldPosition[pId][1] = result[1];
		worldPosition[pId][2] = result[2];

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

		initialPosition.clear();
		d_joints.clear();
		initialNormal.clear();

		all_Nodes.clear();
		all_Meshs.clear();
		nodeId_Dir.clear();
		mesh_Name.clear();
		meshId_Dir.clear();
		node_matrix.clear();

		maxJointId = -1;
		jointNum = -1;
		meshNum = -1;

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
		this->stateRealChannel_1()->clear();
		this->stateTexCoord_0()->clear();
		this->stateTexCoord_1()->clear();
		this->stateWeights_0()->clear();
		this->stateWeights_1()->clear();
		this->stateTextureMesh()->getDataPtr()->clear();

	}



	template<typename TDataType>
	GltfLoader<TDataType>::~GltfLoader()
	{
		InitializationData();

		initialPosition.clear();
		initialNormal.clear();
		d_joints.clear();
	}

	template<typename TDataType>
	void GltfLoader<TDataType>::varRenderChanged()
	{
		jointLineRender->varRadius()->setValue(this->varJointRadius()->getValue() / 2);
		jointPointRender->varPointSize()->setValue(this->varJointRadius()->getValue());
	}


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



	void loadMaterial(tinygltf::Model& model, std::shared_ptr<TextureMesh> texMesh, FilePath filename)
	{
		const std::vector<tinygltf::Material>& sourceMaterials = model.materials;

		auto& reMats = texMesh->materials();
		reMats.clear();
		if (sourceMaterials.size()) //use materials.size()
		{
			reMats.resize(sourceMaterials.size());
		}


		std::vector<tinygltf::Texture>& textures = model.textures;
		std::vector<tinygltf::Image>& images = model.images;
		dyno::CArray2D<dyno::Vec4f> texture(1, 1);
		texture[0, 0] = dyno::Vec4f(1);


		for (int matId = 0; matId < sourceMaterials.size(); matId++)
		{
			auto material = sourceMaterials[matId];
			auto color = material.pbrMetallicRoughness.baseColorFactor;
			auto roughness = material.pbrMetallicRoughness.roughnessFactor;

			auto metallic = material.pbrMetallicRoughness.metallicFactor;

			auto colorTexId = material.pbrMetallicRoughness.baseColorTexture.index;
			auto texCoord = material.pbrMetallicRoughness.baseColorTexture.texCoord;

			reMats[matId] = std::make_shared<Material>();
			reMats[matId]->ambient = { 0,0,0 };
			reMats[matId]->diffuse = Vec3f(color[0], color[1], color[2]);
			reMats[matId]->alpha = color[3];
			reMats[matId]->specular = Vec3f(1 - roughness);
			reMats[matId]->roughness = roughness;

			std::string colorUri = getTexUri(textures, images, colorTexId);

			if (!colorUri.empty())
			{
				auto root = filename.path().parent_path();
				colorUri = (root / colorUri).string();

				if (loadImage(colorUri.c_str(), texture))
				{
					reMats[matId]->texColor.assign(texture);
				}
			}
			else
			{
				if (reMats[matId]->texColor.size())
					reMats[matId]->texColor.clear();
			}

			auto bumpTexId = material.normalTexture.index;
			auto scale = material.normalTexture.scale;
			std::string bumpUri = getTexUri(textures, images, bumpTexId);

			if (!bumpUri.empty())
			{
				auto root = filename.path().parent_path();
				bumpUri = (root / bumpUri).string();

				if (loadImage(bumpUri.c_str(), texture))
				{
					reMats[matId]->texBump.assign(texture);
					reMats[matId]->bumpScale = scale;
				}
			}
			else
			{
				if (reMats[matId]->texBump.size())
					reMats[matId]->texBump.clear();
			}

		}


	}


	template< typename Coord, typename uint, typename Mat4f >
	__global__ void ShapeToCenter(
		DArray<Coord> iniPos,
		DArray<Coord> finalPos,
		DArray<uint> shapeId,
		DArray<Coord> t,
		Mat4f m
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= iniPos.size()) return;

		finalPos[pId] = iniPos[pId] - t[shapeId[pId]];
		Vec4f P = Vec4f(finalPos[pId][0], finalPos[pId][1], finalPos[pId][2], 1);

		P = m * P;
		finalPos[pId] = Coord(P[0], P[1], P[2]);

	}





	DEFINE_CLASS(GltfLoader);
}
