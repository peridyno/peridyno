#pragma once
#include "Array/Array.h"
#include "Topology/TriangleSet.h"
#include "Matrix.h"
#include "Vector.h"

#include "Primitive/Primitive3D.h"
#include "Topology/TextureMesh.h"
#include "tinygltf/tiny_gltf.h"
#include "Field/FilePath.h"
#include "SkinInfo.h"

#define NULL_TIME (-9599.99)

namespace dyno 
{

	typedef unsigned char byte;
	typedef int joint;
	typedef int scene;


	void getBoundingBoxByName(tinygltf::Model& model,const tinygltf::Primitive& primitive,const std::string& attributeName,TAlignedBox3D<Real>& bound,Transform3f& transform);

	void getVec3fByAttributeName(tinygltf::Model& model, const tinygltf::Primitive& primitive, const std::string& attributeName, std::vector<Vec3f>& vertices);

	void getVec4ByAttributeName(tinygltf::Model& model, const tinygltf::Primitive& primitive, const std::string& attributeName, std::vector<Vec4f>& vec4Data);

	void getRealByIndex(tinygltf::Model& model, int index, std::vector<Real>& result);

	void getVec3fByIndex(tinygltf::Model& model, int index, std::vector<Vec3f>& result);

	void getQuatByIndex(tinygltf::Model& model, int index, std::vector<Quat<float>>& result);

	void getTriangles(tinygltf::Model& model, const tinygltf::Primitive& primitive, std::vector<TopologyModule::Triangle>& triangles, int pointOffest);

	std::string getTexUri(const std::vector<tinygltf::Texture>& textures, const std::vector<tinygltf::Image>& images, int index);

	void getVertexBindJoint(tinygltf::Model& model, const tinygltf::Primitive& primitive, const std::string& attributeName, std::vector<Vec4f>& vec4Data, const std::vector<int>& skinJoints);


	void getNodesAndHierarchy(tinygltf::Model& model, std::map<scene, std::vector<int>> Scene_JointsNodesId, std::vector<joint>& all_Nodes, std::map<joint, std::vector<int>>& id_Dir);

	void traverseNode(tinygltf::Model& model, joint id, std::vector<joint>& joint_nodes, std::map<joint, std::vector<int>>& dir, std::vector<joint> currentDir);

	void getJointsTransformData(		
		const std::vector<int>& all_Joints,
		std::vector<std::vector<int>>& joint_child,
		std::map<int, Quat<float>>& joint_rotation,
		std::map<int, Vec3f>& joint_scale,
		std::map<int, Vec3f>& joint_translation,
		std::map<int, Mat4f>& joint_matrix,
		tinygltf::Model model
	);

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
	);

	void buildInverseBindMatrices(
		const std::vector<joint>& all_Joints,
		std::map<joint, Mat4f>& joint_matrix, int& maxJointId,
		tinygltf::Model& model,
		std::map<joint, Quat<float>>& joint_rotation,
		std::map<joint, Vec3f>& joint_translation,
		std::map<joint, Vec3f>& joint_scale,
		std::map<joint, Mat4f>& joint_inverseBindMatrix,
		std::map<joint, std::vector<int>> jointId_joint_Dir
	);

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
		int& maxJointId
	);

	std::vector<int> getJointDirByJointIndex(int Index, std::map<joint, std::vector<int>> jointId_joint_Dir);

	void getMeshMatrix(
		tinygltf::Model& model,
		const std::vector<int>& all_MeshNodeIDs,
		int& maxMeshId,
		CArray<Mat4f>& mesh_Matrix
	);

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
	);

	void loadGLTFTextureMesh(std::shared_ptr<TextureMesh> texMesh, const std::string& filepath);

	void loadGLTFMaterial(tinygltf::Model& model, std::shared_ptr<TextureMesh> texMesh, FilePath filename);

	void loadGLTFShape(
		tinygltf::Model& model, 
		std::shared_ptr<TextureMesh> texMesh, 
		const std::string& filepath,  
		DArray<Vec3f>* initialPosition = nullptr,
		DArray<Vec3f>* initialNormal = nullptr, 
		DArray<Mat4f>* d_mesh_Matrix =nullptr,
		DArray<int>* d_shape_meshId = nullptr,
		std::shared_ptr<SkinInfo> skinData = nullptr
	);



	template< typename Triangle>
	void updateVertexIdShape(
		DArray<Triangle>& triangle,
		DArray<uint>& ID_shapeId,
		int& shapeId,
		int size
	);

	template<typename Mat4f, typename Vec3f >
	void shapeTransform(
		DArray<Vec3f>& intialPosition,
		DArray<Vec3f>& worldPosition,
		DArray<Vec3f>& intialNormal,
		DArray<Vec3f>& Normal,
		DArray<Mat4f>& WorldMatrix,
		DArray<uint>& vertexId_shape,
		DArray<int>& shapeId_MeshId
	);



	template<typename Vec3f>
	void  setupPoints(
		DArray<Vec3f>& newPos,
		DArray<Vec3f>& pos,
		DArray<int>& radix
	);

	template<typename uint>
	void  Shape_PointCounter(
		DArray<int>& counter,
		DArray<uint>& point_ShapeIds,
		uint& target
	);

	template< typename Vec3f, typename uint>
	void shapeToCenter(
		DArray<Vec3f>& iniPos,
		DArray<Vec3f>& finalPos,
		DArray<uint>& shapeId,
		DArray<Vec3f>& t
	);

}
