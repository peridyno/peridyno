#include "GltfLoader.h"
#include <GLPhotorealisticRender.h>

#define NULL_TIME (-9599.99)
#define NULL_POSITION (-959959.9956)
#define TINYGLTF_IMPLEMENTATION

#include "tinygltf/tiny_gltf.h"

#include "GltfFunc.h"
#include "ImageLoader.h"

namespace dyno
{

	IMPLEMENT_CLASS(BoundingBoxOfTextureMesh);

	BoundingBoxOfTextureMesh::BoundingBoxOfTextureMesh()
		: ComputeModule()
	{
		auto callback = std::make_shared<FCallBackFunc>(std::bind(&BoundingBoxOfTextureMesh::shapeIdChanged, this));

		this->varShapeId()->attach(callback);
	}

	void BoundingBoxOfTextureMesh::compute()
	{
		shapeIdChanged();
	}

	void BoundingBoxOfTextureMesh::shapeIdChanged()
	{
		uint shapeId = this->varShapeId()->getValue();
		auto mesh = this->inTextureMesh()->constDataPtr();

		if (mesh == nullptr)
		{
			return;
		}

		if (this->outBoundingBox()->isEmpty())
		{
			this->outBoundingBox()->allocate();
		}

		auto& es = this->outBoundingBox()->getData();

		if (shapeId < mesh->shapes().size())
		{
			auto bb = mesh->shapes()[shapeId]->boundingBox;

			this->varCenter()->setValue((bb.v0 + bb.v1) / 2);
			this->varLowerBound()->setValue(bb.v0);
			this->varUpperBound()->setValue(bb.v1);

			std::vector<Vec3f> vertices;

			Vec3f v0 = bb.v0;
			Vec3f v1 = Vec3f(bb.v0.x, bb.v0.y, bb.v1.z);
			Vec3f v2 = Vec3f(bb.v1.x, bb.v0.y, bb.v1.z);
			Vec3f v3 = Vec3f(bb.v1.x, bb.v0.y, bb.v0.z);

			Vec3f v4 = Vec3f(bb.v0.x, bb.v1.y, bb.v0.z);
			Vec3f v5 = Vec3f(bb.v0.x, bb.v1.y, bb.v1.z);
			Vec3f v6 = bb.v1;
			Vec3f v7 = Vec3f(bb.v1.x, bb.v1.y, bb.v0.z);

			vertices.push_back(v0);
			vertices.push_back(v1);
			vertices.push_back(v2);
			vertices.push_back(v3);
			vertices.push_back(v4);
			vertices.push_back(v5);
			vertices.push_back(v6);
			vertices.push_back(v7);

			std::vector<TopologyModule::Edge> edges;
			edges.push_back(TopologyModule::Edge(0, 1));
			edges.push_back(TopologyModule::Edge(1, 2));
			edges.push_back(TopologyModule::Edge(2, 3));
			edges.push_back(TopologyModule::Edge(3, 0));

			edges.push_back(TopologyModule::Edge(4, 5));
			edges.push_back(TopologyModule::Edge(5, 6));
			edges.push_back(TopologyModule::Edge(6, 7));
			edges.push_back(TopologyModule::Edge(7, 4));

			edges.push_back(TopologyModule::Edge(0, 4));
			edges.push_back(TopologyModule::Edge(1, 5));
			edges.push_back(TopologyModule::Edge(2, 6));
			edges.push_back(TopologyModule::Edge(3, 7));

			es.setPoints(vertices);
			es.setEdges(edges);

			es.update();
		}
		else
			es.clear();
	}

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
		this->varUseInstanceTransform()->attach(callbackTransform);


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

		this->stateAnimation()->setDataPtr(std::make_shared<JointAnimationInfo>());
		this->stateAnimation()->promoteOuput();

		auto glShapeCenter = std::make_shared<GLPointVisualModule>();
		glShapeCenter->setColor(Color(1.0f, 1.0f, 0.0f));
		glShapeCenter->varPointSize()->setValue(this->varJointRadius()->getValue() * 2);
		glShapeCenter->setVisible(true);
		this->stateShapeCenter()->connect(glShapeCenter->inPointSet());
		this->graphicsPipeline()->pushModule(glShapeCenter);

		auto showBoundingBox = std::make_shared<BoundingBoxOfTextureMesh>();
		this->stateTextureMesh()->connect(showBoundingBox->inTextureMesh());
		this->graphicsPipeline()->pushModule(showBoundingBox);

		auto bbRender = std::make_shared<GLWireframeVisualModule>();
		showBoundingBox->outBoundingBox()->connect(bbRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(bbRender);

		this->stateSkin()->setDataPtr(std::make_shared<SkinInfo>());
		this->stateJointsData()->setDataPtr(std::make_shared<JointInfo>());

		this->stateTextureMesh()->promoteOuput();

		this->setForceUpdate(false);

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


		TinyGLTF loader;
		std::string err;
		std::string warn;
		std::string filename = this->varFileName()->getValue().string();


		bool ret = loader.LoadASCIIFromFile(newModel, &err, &warn, filename);

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
		importAnimation(*newModel, joint_output, joint_input, joint_T_f_anim, joint_T_Time, joint_S_f_anim, joint_S_Time, joint_R_f_anim, joint_R_Time);

		for (int i = 0; i< newModel->nodes.size();i++)
		{
			node_Name[i] = newModel->nodes[i].name;
		}

		// import Scenes:

		std::map<scene, std::vector<int>> Scene_Nodes;
		for (size_t i = 0; i < newModel->scenes.size(); i++)
		{
			std::vector<int> vecS_Roots;
			vecS_Roots = newModel->scenes[i].nodes;
			Scene_Nodes[i] = vecS_Roots;
		}
		std::vector<int> all_Nodes;
		std::map<joint, std::vector<int>> nodeId_Dir;
		int jointNum = -1;
		int meshNum = -1;
		std::vector<int> all_Meshs;
		std::map<int, std::vector<int>> meshId_Dir;

		getNodesAndHierarchy(*newModel, Scene_Nodes, all_Nodes, nodeId_Dir);	//jointId_joint_Dir	//update private: std::map<joint, std::vector<int>> jointId_joint_Dir;

		updateJoint_Mesh_Camera_Dir(*newModel, jointNum, meshNum, jointId_joint_Dir, all_Joints, all_Nodes, nodeId_Dir, meshId_Dir, all_Meshs, maxJointId);

		std::vector<std::vector<int>> joint_child;	//build edgeset;

		//get Local Transform T S R M 
		getJointsTransformData(all_Nodes, joint_child, joint_rotation, joint_scale, joint_translation, joint_matrix, *newModel);


		d_joints.assign(all_Joints);

		//get InverseBindMatrix (Global)
		this->buildInverseBindMatrices(all_Joints);

		std::vector<Mat4f> localMatrix;
		localMatrix.resize(maxJointId + 1);

		for (auto jId : all_Joints)
		{
			localMatrix[jId] = joint_matrix[jId];
		}

		this->stateJointLocalMatrix()->assign(localMatrix);


		// get joint World Location	
		printf("************  Set Joint  ************\n");
		{
			std::vector<Coord> jointVertices;
			std::map<int, int> jointId_VId;

			for (size_t j = 0; j < jointNum; j++)
			{
				joint jId = all_Joints[j];
				jointId_VId[jId] = jointVertices.size();

				jointVertices.push_back(getVertexLocationWithJointTransform(jId, Vec3f(0, 0, 0), joint_matrix));

			}

			//
			this->stateJointSet()->getDataPtr()->setPoints(jointVertices);
			std::vector<TopologyModule::Edge> edges;

			for (size_t i = 0; i < jointNum; i++)
			{
				for (auto childId : joint_child[all_Joints[i]])
				{
					edges.push_back(TopologyModule::Edge(i, jointId_VId[childId]));
				}
			}
			this->stateJointSet()->getDataPtr()->setEdges(edges);

			jointVertices.clear();
		}

		auto texMesh = this->stateTextureMesh()->getDataPtr();
		//materials
		loadGLTFMaterial(*newModel, texMesh, filename);

		
		loadGLTFShape(*newModel, texMesh, filename, &initialPosition,&initialNormal, &d_mesh_Matrix,&d_shape_meshId, this->stateSkin()->getDataPtr());
		

		this->updateTransform();
		
		this->stateSkin()->getDataPtr()->mesh = texMesh;

		this->stateSkin()->getDataPtr()->initialPosition = initialPosition;

		this->stateSkin()->getDataPtr()->initialNormal = initialNormal;

		this->updateAnimation(0);

		this->stateJointsData()->getDataPtr()->UpdateJointInfo(
			this->stateJointInverseBindMatrix()->getData(),
			this->stateJointLocalMatrix()->getData(),
			this->stateJointWorldMatrix()->getData(),
			all_Joints,
			jointId_joint_Dir,
			joint_translation,
			joint_scale,
			joint_rotation
		);

		this->stateJointsData()->getDataPtr()->setJointName(joint_Name);

		this->stateAnimation()->getDataPtr()->setAnimationData(
			joint_T_f_anim,
			joint_T_Time,
			joint_S_f_anim,
			joint_S_Time,
			joint_R_f_anim,
			joint_R_Time,
			this->stateJointsData()->getDataPtr()
			);

		delete newModel;
	}

	// ***************************** function *************************** //

	template<typename TDataType>
	void GltfLoader<TDataType>::updateTransform()
	{
		//updateModelTransformMatrix
		this->updateTransformState();

		if (all_Joints.size())	//Animation
		{
			if (varImportAnimation()->getValue() && (!this->joint_R_f_anim.empty() && !this->joint_T_f_anim.empty() && !this->joint_S_f_anim.empty()))
				updateAnimation(this->stateFrameNumber()->getValue());

		}

		if (this->stateTextureMesh()->getDataPtr()->vertices().size())
		{
			//Move by Dir
			if (true)
			{
				cuExecute(this->stateTextureMesh()->getDataPtr()->vertices().size(),
					ShapeTransform,
					initialPosition,
					this->stateTextureMesh()->getDataPtr()->vertices(),
					initialNormal,
					this->stateTextureMesh()->getDataPtr()->normals(),
					d_mesh_Matrix,
					this->stateTextureMesh()->getDataPtr()->shapeIds(),
					d_shape_meshId
				);
			}

			//Move by VarTransform
		}



		//update BoundingBox 

		auto shapeNum = this->stateTextureMesh()->getDataPtr()->shapes().size();

		CArray<Coord> c_shapeCenter;
		c_shapeCenter.resize(shapeNum);
		//counter
		for (uint i = 0; i < shapeNum; i++)
		{
			DArray<int> counter;
			counter.resize(this->stateTextureMesh()->getDataPtr()->vertices().size());


			cuExecute(this->stateTextureMesh()->getDataPtr()->vertices().size(),
				C_Shape_PointCounter,
				counter,
				this->stateTextureMesh()->getDataPtr()->shapeIds(),
				i
			);

			Reduction<int> reduce;
			int num = reduce.accumulate(counter.begin(), counter.size());

			DArray<Coord> targetPoints;
			targetPoints.resize(num);

			Scan<int> scan;
			scan.exclusive(counter.begin(), counter.size());

			cuExecute(this->stateTextureMesh()->getDataPtr()->vertices().size(),
				C_SetupPoints,
				targetPoints,
				this->stateTextureMesh()->getDataPtr()->vertices(),
				counter
			);


			Reduction<Coord> reduceBounding;

			auto& bounding = this->stateTextureMesh()->getDataPtr()->shapes()[i]->boundingBox;
			Coord lo = reduceBounding.minimum(targetPoints.begin(), targetPoints.size());
			Coord hi = reduceBounding.maximum(targetPoints.begin(), targetPoints.size());

			bounding.v0 = lo;
			bounding.v1 = hi;
			this->stateTextureMesh()->getDataPtr()->shapes()[i]->boundingTransform.translation() = (lo + hi) / 2;

			c_shapeCenter[i] = (lo + hi) / 2;

			targetPoints.clear();

			counter.clear();
		}

		d_ShapeCenter.assign(c_shapeCenter);	// Used to "ToCenter"
		unCenterPosition.assign(this->stateTextureMesh()->getDataPtr()->vertices());

		//ToCenter
		if (varUseInstanceTransform()->getValue())
		{
			cuExecute(this->stateTextureMesh()->getDataPtr()->vertices().size(),
				ShapeToCenter,
				unCenterPosition,
				this->stateTextureMesh()->getDataPtr()->vertices(),
				this->stateTextureMesh()->getDataPtr()->shapeIds(),
				d_ShapeCenter
			);

			auto& reShapes = this->stateTextureMesh()->getDataPtr()->shapes();

			for (size_t i = 0; i < shapeNum; i++)
			{
				reShapes[i]->boundingTransform.translation() = reShapes[i]->boundingTransform.translation() + this->varLocation()->getValue();
			}
		}
		else
		{
			auto& reShapes = this->stateTextureMesh()->getDataPtr()->shapes();

			for (size_t i = 0; i < shapeNum; i++)
			{
				reShapes[i]->boundingTransform.translation() = Vec3f(0);
			}
		}

		this->stateShapeCenter()->getDataPtr()->setPoints(d_ShapeCenter);

	}

	template<typename TDataType>
	void GltfLoader<TDataType>::updateStates()
	{
		ParametricModel<TDataType>::updateStates();

		if (joint_output.empty() || !this->varImportAnimation()->getValue() || (this->joint_R_f_anim.empty()&&this->joint_T_f_anim.empty()&&this->joint_S_f_anim.empty()))
			return;

		updateAnimation(this->stateFrameNumber()->getValue());
		auto jointInfo = this->stateJointsData()->getDataPtr();
		
		this->stateJointsData()->getDataPtr()->UpdateJointInfo(
			this->stateJointInverseBindMatrix()->getData(), 
			this->stateJointLocalMatrix()->getData(), 
			this->stateJointWorldMatrix()->getData(),
			all_Joints,
			jointId_joint_Dir,
			joint_translation,
			joint_scale,
			joint_rotation
		);
	}; 


	template<typename TDataType>
	void GltfLoader<TDataType>::updateAnimation(int frameNumber)
	{
		if (joint_output.empty() || all_Joints.empty() || joint_matrix.empty())
			return;

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

		auto& skinInfo = this->stateSkin()->getData();


		for (size_t i = 0; i < skinInfo.size(); i++)//
		{
			auto& bindJoint0 = skinInfo.V_jointID_0[i];
			auto& bindJoint1 = skinInfo.V_jointID_1[i];

			auto& bindWeight0 = skinInfo.V_jointWeight_0[i];
			auto& bindWeight1 = skinInfo.V_jointWeight_1[i];

			for (size_t j = 0; j < skin_VerticeRange[i].size(); j++)
			{
				//
				Vec2u& range = skinInfo.skin_VerticeRange[i][j];

				skinAnimation(initialPosition,
					mesh->vertices(),
					this->stateJointInverseBindMatrix()->getData(),
					this->stateJointWorldMatrix()->getData(),

					bindJoint0,
					bindJoint1,
					bindWeight0,
					bindWeight1,
					this->stateTransform()->getValue(),
					false,
					range
				);

				//update Normals

				skinAnimation(
					initialNormal,
					mesh->normals(),
					this->stateJointInverseBindMatrix()->getData(),
					this->stateJointWorldMatrix()->getData(),

					bindJoint0,
					bindJoint1,
					bindWeight0,
					bindWeight1,
					this->stateTransform()->getValue(),
					true,
					range
				);

			}
		}
		

	};


	template<typename TDataType>
	void GltfLoader<TDataType>::updateAnimationMatrix(const std::vector<joint>& all_Joints, int currentframe)
	{
		joint_AnimaMatrix = joint_matrix;

		for (auto jId : all_Joints)
		{
			const std::vector<int>& jD = getJointDirByJointIndex(jId,jointId_joint_Dir);

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

		const std::vector<int>& jD =  getJointDirByJointIndex(jointId, jointId_joint_Dir);

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


		for (size_t i = 0; i < allJoints.size(); i++)
		{
			joint jointId = allJoints[i];
			const std::vector<int>& jD = getJointDirByJointIndex(jointId, jointId_joint_Dir);



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
			//temp.push_back(Mat4f::identityMatrix());
			temp[i] = Mat4f::identityMatrix();

		}



		for (size_t i = 0; i < all_Joints.size(); i++)
		{
			joint jointId = all_Joints[i];

			const std::vector<int>& jD = getJointDirByJointIndex(jointId, jointId_joint_Dir);

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
		if (pId == 1) 
		{
			auto iP = intialPosition[pId];
		}

		tempV = WorldMatrix[MeshId] * tempV;
		tempN = WorldMatrix[MeshId] * tempN;

		worldPosition[pId] = Coord(tempV[0], tempV[1], tempV[2]);
		Normal[pId] = Coord(tempN[0], tempN[1], tempN[2]);
		if (pId == 1)
		{
			auto iP = worldPosition[pId];
		}
		
	}


	



	template< typename Coord, typename Vec4f, typename Mat4f ,typename Vec2u>
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

		initialPosition.clear();
		d_joints.clear();
		initialNormal.clear();


		meshId_Dir.clear();
		node_matrix.clear();

		d_mesh_Matrix.clear();
		d_shape_meshId.clear();
		unCenterPosition.clear();
		skin_VerticeRange.clear();
		node_Name.clear();


		maxJointId = -1;



		this->stateInitialMatrix()->clear();
		this->stateJointInverseBindMatrix()->clear();
		this->stateJointLocalMatrix()->clear();

		this->stateJointWorldMatrix()->clear();
		this->stateTexCoord_0()->clear();
		this->stateTexCoord_1()->clear();
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



	template< typename Coord, typename uint>
	__global__ void ShapeToCenter(
		DArray<Coord> iniPos,
		DArray<Coord> finalPos,
		DArray<uint> shapeId,
		DArray<Coord> t
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= iniPos.size()) return;

		finalPos[pId] = iniPos[pId] - t[shapeId[pId]];
		Vec4f P = Vec4f(finalPos[pId][0], finalPos[pId][1], finalPos[pId][2], 1);

		finalPos[pId] = Coord(P[0], P[1], P[2]);

	}



	template< typename Coord, typename uint>
	__global__ void initialCenterCoord(
		DArray<Coord> Pos,
		DArray<uint> shapeId,
		DArray<int> pointId,
		DArray<Coord> iniPoint,
		int shapeNum
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Pos.size()) return;
	
		for (int i = 0; i < shapeNum; i++)
		{
			if (shapeId[pId] == i && pointId[i] == -1)
			{
				pointId[i] = i;
				iniPoint[i] = Pos[pId];
				break;
			}
		}
	}


	template<typename uint>
	__global__ void  C_Shape_PointCounter(
		DArray<int> counter,
		DArray<uint> point_ShapeIds,
		uint target
		)
	{
		uint tId = threadIdx.x + blockDim.x * blockIdx.x;
		if (tId >= point_ShapeIds.size()) return;

		counter[tId] = (point_ShapeIds[tId]== target) ? 1 : 0;
	}


	template<typename Coord>
	__global__ void  C_SetupPoints(
		DArray<Coord> newPos,
		DArray<Coord> pos,
		DArray<int> radix
	)
	{
		uint tId = threadIdx.x + blockDim.x * blockIdx.x;
		if (tId >= pos.size()) return;

		if (tId < pos.size() - 1 && radix[tId] != radix[tId + 1])
		{
			newPos[radix[tId]] = pos[tId];
		}
		else if (tId == pos.size() - 1 && pos.size() > 2)
		{
			if(radix[tId] != radix[tId - 1])
				newPos[radix[tId]] = pos[tId];
		}
		
	}


	DEFINE_CLASS(GltfLoader);
}
