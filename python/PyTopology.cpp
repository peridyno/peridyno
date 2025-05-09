#include "PyTopology.h"

#include <Collision/CollisionDetectionAlgorithm.h>
#include <Topology/HierarchicalModel.h>

#include "Topology/TextureMesh.h"
void declare_texture_mesh(py::module& m) {
	using Class = dyno::TextureMesh;
	using Parent = dyno::TopologyModule;
	std::string pyclass_name = std::string("TextureMesh");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("vertices", &Class::vertices)
		.def("normals", &Class::normals)
		.def("tex_coords", &Class::texCoords)
		.def("shape_ids", &Class::shapeIds)

		.def("shapes", &Class::shapes)
		.def("materials", &Class::materials)
		.def("merge", &Class::merge)
		.def("clear", &Class::clear);
}

void declare_attribute(py::module& m) {
	using Class = dyno::Attribute;
	std::string pyclass_name = std::string("Attribute");
	py::class_<Class, std::shared_ptr<Class>>Attribute(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	Attribute.def(py::init<>())
		.def("set_material_type", &Class::setMaterialType)
		.def("set_kinematic_type", &Class::setKinematicType)
		.def("set_object_id", &Class::setObjectId)

		.def("material_type", &Class::materialType)
		.def("kinematic_type", &Class::kinematicType)

		.def("is_fluid", &Class::isFluid)
		.def("is_rigid", &Class::isRigid)
		.def("is_elastic", &Class::isElastic)
		.def("is_plastic", &Class::isPlastic)

		.def("set_fluid", &Class::setFluid)
		.def("set_rigid", &Class::setRigid)
		.def("set_elastic", &Class::setElastic)
		.def("set_plastic", &Class::setPlastic)

		.def("is_fixed", &Class::isFixed)
		.def("is_passive", &Class::isPassive)
		.def("is_dynamic", &Class::isDynamic)

		.def("set_fixed", &Class::setFixed)
		.def("set_passive", &Class::setPassive)
		.def("set_dynamic", &Class::setDynamic)

		.def("object_id", &Class::objectId);

	py::enum_<typename Class::KinematicType>(Attribute, "KinematicType")
		.value("KINEMATIC_MASK", Class::KinematicType::KINEMATIC_MASK)
		.value("KINEMATIC_FIXED", Class::KinematicType::KINEMATIC_FIXED)
		.value("KINEMATIC_PASSIVE", Class::KinematicType::KINEMATIC_PASSIVE)
		.value("KINEMATIC_POSITIVE", Class::KinematicType::KINEMATIC_POSITIVE)
		.export_values();

	py::enum_<typename Class::MaterialType>(Attribute, "MaterialType")
		.value("MATERIAL_MASK", Class::MaterialType::MATERIAL_MASK)
		.value("MATERIAL_FLUID", Class::MaterialType::MATERIAL_FLUID)
		.value("MATERIAL_RIGID", Class::MaterialType::MATERIAL_RIGID)
		.value("MATERIAL_ELASTIC", Class::MaterialType::MATERIAL_ELASTIC)
		.value("MATERIAL_PLASTIC", Class::MaterialType::MATERIAL_PLASTIC)
		.export_values();

	py::enum_<typename Class::ObjectID>(Attribute, "ObjectID")
		.value("OBJECTID_MASK", Class::ObjectID::OBJECTID_MASK)
		.value("OBJECTID_INVALID", Class::ObjectID::OBJECTID_INVALID)
		.export_values();
}

void pybind_topology(py::module& m)
{
	// Collision
	declare_attribute(m);
	declare_calculate_bounding_box<dyno::DataType3f>(m, "3f");
	declare_collision_detection_broad_phase<dyno::DataType3f>(m, "3f");
	declare_collistion_detection_bounding_box<dyno::DataType3f>(m, "3f");
	declare_collistion_detection_triangle_set<dyno::DataType3f>(m, "3f");
	declare_neighbor_element_query<dyno::DataType3f>(m, "3f");
	declare_neighbor_point_query<dyno::DataType3f>(m, "3f");
	declare_neighbor_triangle_query<dyno::DataType3f>(m, "3f");

	// Mapping
	declare_anchor_point_to_point_set<dyno::DataType3f>(m, "3f");
	declare_bounding_box_to_edge_set<dyno::DataType3f>(m, "3f");
	declare_contacts_to_edge_set<dyno::DataType3f>(m, "3f");
	declare_contacts_to_point_set<dyno::DataType3f>(m, "3f");
	declare_discrete_elements_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_extract_edge_set_from_polygon_set<dyno::DataType3f>(m, "3f");
	declare_extract_triangle_set_from_polygon_set<dyno::DataType3f>(m, "3f");
	declare_extract_qaud_set_from_polygon_set<dyno::DataType3f>(m, "3f");
	declare_frame_to_point_set<dyno::DataType3f>(m, "3f");
	declare_height_field_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_merge_simplex_set<dyno::DataType3f>(m, "3f");
	declare_merge_triangle_set<dyno::DataType3f>(m, "3f");
	declare_point_set_to_point_set<dyno::DataType3f>(m, "3f");
	declare_point_set_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_quad_set_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_split_simplex_set<dyno::DataType3f>(m, "3f");
	declare_tetrahedron_set_to_point_set<dyno::DataType3f>(m, "3f");
	declare_texture_mesh_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_texture_mesh_to_triangle_set_node<dyno::DataType3f>(m, "3f");

	// Module
	declare_calculate_maximum<dyno::DataType3f>(m, "3f");
	declare_calculate_minimum<dyno::DataType3f>(m, "3f");
	declare_calculate_norm<dyno::DataType3f>(m, "3f");

	py::class_<dyno::PdActor, std::shared_ptr<dyno::PdActor>>(m, "PdActor")
		.def(py::init<>())
		.def_readwrite("idx", &dyno::PdActor::idx)
		.def_readwrite("shapeType", &dyno::PdActor::shapeType)
		.def_readwrite("center", &dyno::PdActor::center)
		.def_readwrite("rot", &dyno::PdActor::rot);

	// Topology
	declare_animation_curve<dyno::DataType3f>(m, "3f");

	declare_joint<float>(m, "f");
	declare_ball_and_socket_joint<float>(m, "f");
	declare_slider_joint<float>(m, "f");
	declare_hinge_joint<float>(m, "f");
	declare_fixed_joint<float>(m, "f");
	declare_point_joint<float>(m, "f");
	declare_distance_joint<float>(m, "f");
	declare_discrete_elements<dyno::DataType3f>(m, "3f");
	declare_distance_field3D<dyno::DataType3f>(m, "3f");
	declare_texture_mesh(m);
	declare_point_set<dyno::DataType3f>(m, "3f");
	declare_edge_set<dyno::DataType3f>(m, "3f");
	declare_frame<dyno::DataType3f>(m, "3f");
	declare_grid_hash<dyno::DataType3f>(m, "3f");
	declare_grid_set<dyno::DataType3f>(m, "3f");
	declare_height_field<dyno::DataType3f>(m, "3f");
	declare_quad_set<dyno::DataType3f>(m, "3f");
	declare_hexahedron_set<dyno::DataType3f>(m, "3f");
	//declare_joint_tree<dyno::DataType3f>(m, "3f");
	declare_level_set<dyno::DataType3f>(m, "3f");
	declare_linear_bvh<dyno::DataType3f>(m, "3f");
	declare_polygon_set<dyno::DataType3f>(m, "3f");
	declare_simplex_set<dyno::DataType3f>(m, "3f");
	declare_sparse_grid_hash<dyno::DataType3f>(m, "3f");
	declare_sparse_octree<dyno::DataType3f>(m, "3f");
	declare_structured_point_set<dyno::DataType3f>(m, "3f");
	declare_triangle_set<dyno::DataType3f>(m, "3f");
	declare_tetrahedron_set<dyno::DataType3f>(m, "3f");
	declare_uniform_grid3D<dyno::DataType3f>(m, "3f");
	declare_unstructured_point_set<dyno::DataType3f>(m, "3f");

	py::enum_<typename dyno::CModeMask>(m, "CModeMask")
		.value("CM_Disabled", dyno::CModeMask::CM_Disabled)
		.value("CM_OriginDCD_Tet", dyno::CModeMask::CM_OriginDCD_Tet)
		.value("CM_InputSDF_Tet", dyno::CModeMask::CM_InputSDF_Tet)
		.value("CM_RigidSurface_Tet", dyno::CModeMask::CM_RigidSurface_Tet)
		.value("CM_TetMesh_Tet", dyno::CModeMask::CM_TetMesh_Tet)
		.value("CM_SurfaceMesh_Tet", dyno::CModeMask::CM_SurfaceMesh_Tet)
		.value("CM_OriginDCD_Sphere", dyno::CModeMask::CM_OriginDCD_Sphere)
		.value("CM_InputSDF_Sphere", dyno::CModeMask::CM_InputSDF_Sphere);

	py::enum_<typename dyno::BodyType>(m, "BodyType")
		.value("Static", dyno::BodyType::Static)
		.value("Kinematic", dyno::BodyType::Kinematic)
		.value("Dynamic", dyno::BodyType::Dynamic)
		.value("NonRotatable", dyno::BodyType::NonRotatable)
		.value("NonGravitative", dyno::BodyType::NonGravitative);

	py::enum_<typename dyno::CollisionMask>(m, "CollisionMask")
		.value("CT_AllObjects", dyno::CollisionMask::CT_AllObjects)
		.value("CT_BoxExcluded", dyno::CollisionMask::CT_BoxExcluded)
		.value("CT_TetExcluded", dyno::CollisionMask::CT_TetExcluded)
		.value("CT_CapsuleExcluded", dyno::CollisionMask::CT_CapsuleExcluded)
		.value("CT_SphereExcluded", dyno::CollisionMask::CT_SphereExcluded)
		.value("CT_BoxOnly", dyno::CollisionMask::CT_BoxOnly)
		.value("CT_TetOnly", dyno::CollisionMask::CT_TetOnly)
		.value("CT_CapsuleOnly", dyno::CollisionMask::CT_CapsuleOnly)
		.value("CT_SphereOnly", dyno::CollisionMask::CT_SphereOnly)
		.value("CT_Disabled", dyno::CollisionMask::CT_Disabled);

	py::enum_<typename dyno::ContactType>(m, "ContactType")
		.value("CT_BOUDNARY", dyno::ContactType::CT_BOUDNARY)
		.value("CT_INTERNAL", dyno::ContactType::CT_INTERNAL)
		.value("CT_NONPENETRATION", dyno::ContactType::CT_NONPENETRATION)
		.value("CT_UNKNOWN", dyno::ContactType::CT_UNKNOWN);

	py::enum_<typename dyno::ConstraintType>(m, "ConstraintType")
		.value("CN_NONPENETRATION", dyno::ConstraintType::CN_NONPENETRATION)
		.value("CN_FRICTION", dyno::ConstraintType::CN_FRICTION)
		.value("CN_FLUID_STICKINESS", dyno::ConstraintType::CN_FLUID_STICKINESS)
		.value("CN_FLUID_SLIPINESS", dyno::ConstraintType::CN_FLUID_SLIPINESS)
		.value("CN_FLUID_NONPENETRATION", dyno::ConstraintType::CN_FLUID_NONPENETRATION)
		.value("CN_GLOBAL_NONPENETRATION", dyno::ConstraintType::CN_GLOBAL_NONPENETRATION)
		.value("CN_LOACL_NONPENETRATION", dyno::ConstraintType::CN_LOACL_NONPENETRATION)
		.value("CN_ANCHOR_EQUAL_1", dyno::ConstraintType::CN_ANCHOR_EQUAL_1)
		.value("CN_ANCHOR_EQUAL_2", dyno::ConstraintType::CN_ANCHOR_EQUAL_2)
		.value("CN_ANCHOR_EQUAL_3", dyno::ConstraintType::CN_ANCHOR_EQUAL_3)
		.value("CN_ANCHOR_TRANS_1", dyno::ConstraintType::CN_ANCHOR_TRANS_1)
		.value("CN_ANCHOR_TRANS_2", dyno::ConstraintType::CN_ANCHOR_TRANS_2)
		.value("CN_BAN_ROT_1", dyno::ConstraintType::CN_BAN_ROT_1)
		.value("CN_BAN_ROT_2", dyno::ConstraintType::CN_BAN_ROT_2)
		.value("CN_BAN_ROT_3", dyno::ConstraintType::CN_BAN_ROT_3)
		.value("CN_ALLOW_ROT1D_1", dyno::ConstraintType::CN_ALLOW_ROT1D_1)
		.value("CN_ALLOW_ROT1D_2", dyno::ConstraintType::CN_ALLOW_ROT1D_2)
		.value("CN_JOINT_SLIDER_MIN", dyno::ConstraintType::CN_JOINT_SLIDER_MIN)
		.value("CN_JOINT_SLIDER_MAX", dyno::ConstraintType::CN_JOINT_SLIDER_MAX)
		.value("CN_JOINT_SLIDER_MOTER", dyno::ConstraintType::CN_JOINT_SLIDER_MOTER)
		.value("CN_JOINT_HINGE_MIN", dyno::ConstraintType::CN_JOINT_HINGE_MIN)
		.value("CN_JOINT_HINGE_MAX", dyno::ConstraintType::CN_JOINT_HINGE_MAX)
		.value("CN_JOINT_HINGE_MOTER", dyno::ConstraintType::CN_JOINT_HINGE_MOTER)
		.value("CN_JOINT_NO_MOVE_1", dyno::ConstraintType::CN_JOINT_NO_MOVE_1)
		.value("CN_JOINT_NO_MOVE_2", dyno::ConstraintType::CN_JOINT_NO_MOVE_2)
		.value("CN_JOINT_NO_MOVE_3", dyno::ConstraintType::CN_JOINT_NO_MOVE_3)
		.value("CN_UNKNOWN", dyno::ConstraintType::CN_UNKNOWN);

	py::enum_<typename dyno::SeparationType>(m, "SeparationType")
		.value("CT_POINT", dyno::SeparationType::CT_POINT)
		.value("CT_EDGE", dyno::SeparationType::CT_EDGE)
		.value("CT_TRIA", dyno::SeparationType::CT_TRIA)
		.value("CT_TRIB", dyno::SeparationType::CT_TRIB)
		.value("CT_RECTA", dyno::SeparationType::CT_RECTA)
		.value("CT_RECTB", dyno::SeparationType::CT_RECTB);

	py::enum_<typename dyno::ElementType>(m, "ElementType")
		.value("ET_BOX", dyno::ElementType::ET_BOX)
		.value("ET_TET", dyno::ElementType::ET_TET)
		.value("ET_CAPSULE", dyno::ElementType::ET_CAPSULE)
		.value("ET_SPHERE", dyno::ElementType::ET_SPHERE)
		.value("ET_TRI", dyno::ElementType::ET_TRI)
		.value("ET_Other", dyno::ElementType::ET_Other)
		.export_values();

	py::class_<dyno::EKey, std::shared_ptr<dyno::EKey>>(m, "EKey")
		.def(py::init<>())
		.def("is_valid", &dyno::EKey::isValid);

	py::class_<dyno::BVHNode, std::shared_ptr<dyno::BVHNode>>(m, "BVHNode")
		.def(py::init<>())
		.def("is_leaf", &dyno::BVHNode::isLeaf);

	py::class_<dyno::QKey, std::shared_ptr<dyno::QKey>>(m, "QKey")
		.def(py::init<>());

	py::class_<dyno::OctreeNode, std::shared_ptr<dyno::OctreeNode>>(m, "OctreeNode")
		.def(py::init<>())
		.def(py::init<dyno::OcKey>())
		.def(py::init<dyno::Level, dyno::OcIndex, dyno::OcIndex, dyno::OcIndex>())
		//.def("is_contained_in", &dyno::OctreeNode::isContainedIn)
		.def("is_contained_strictly_in", &dyno::OctreeNode::isContainedStrictlyIn)

		.def("get_coord", &dyno::OctreeNode::getCoord)

		.def("least_common_ancestor", &dyno::OctreeNode::leastCommonAncestor)
		.def("key", &dyno::OctreeNode::key)
		.def("level", &dyno::OctreeNode::level)
		.def("set_data_index", &dyno::OctreeNode::setDataIndex)
		.def("get_data_index", &dyno::OctreeNode::getDataIndex)
		.def("set_first_child_index", &dyno::OctreeNode::setFirstChildIndex)
		.def("get_first_child_index", &dyno::OctreeNode::getFirstChildIndex)
		.def("set_data_size", &dyno::OctreeNode::setDataSize)
		.def("get_data_size", &dyno::OctreeNode::getDataSize)
		.def("is_valid", &dyno::OctreeNode::isValid)
		.def("is_empty", &dyno::OctreeNode::isEmpty)

		.def_readwrite("m_key", &dyno::OctreeNode::m_key)
		.def_readwrite("m_level", &dyno::OctreeNode::m_level)
		.def_readwrite("m_data_loc", &dyno::OctreeNode::m_data_loc)
		.def_readwrite("m_start_loc", &dyno::OctreeNode::m_start_loc)
		.def_readwrite("m_data_size", &dyno::OctreeNode::m_data_size)
		.def_readwrite("m_current_loc", &dyno::OctreeNode::m_current_loc)
		.def_readwrite("m_first_child_loc", &dyno::OctreeNode::m_first_child_loc)
		.def_readwrite("m_bCopy", &dyno::OctreeNode::m_bCopy);

	py::class_<dyno::Material, dyno::Object, std::shared_ptr<dyno::Material>>(m, "Material")
		.def(py::init<>())
		.def_readwrite("base_color", &dyno::Material::baseColor)
		.def_readwrite("metallic", &dyno::Material::metallic)
		.def_readwrite("roughness", &dyno::Material::roughness)
		.def_readwrite("alpha", &dyno::Material::alpha)
		.def_readwrite("tex_color", &dyno::Material::texColor)
		.def_readwrite("tex_bump", &dyno::Material::texBump)
		.def_readwrite("bump_scale", &dyno::Material::bumpScale);

	py::class_<dyno::Shape, dyno::Object, std::shared_ptr<dyno::Shape>>(m, "Shape")
		.def(py::init<>())
		.def_readwrite("vertex_index", &dyno::Shape::vertexIndex)
		.def_readwrite("normal_index", &dyno::Shape::normalIndex)
		.def_readwrite("texCoord_index", &dyno::Shape::texCoordIndex)
		.def_readwrite("bounding_box", &dyno::Shape::boundingBox)
		.def_readwrite("bounding_transform", &dyno::Shape::boundingTransform)
		.def_readwrite("material", &dyno::Shape::material);

	py::class_<dyno::TKey, std::shared_ptr<dyno::TKey>>(m, "TKey")
		.def(py::init<>())
		.def(py::init<dyno::PointType, dyno::PointType, dyno::PointType>());

	py::class_<dyno::ModelObject, dyno::Object, std::shared_ptr<dyno::ModelObject>>(m, "ModelObject")
		.def(py::init<>())
		.def_readwrite("name", &dyno::ModelObject::name)
		.def_readwrite("localTransform", &dyno::ModelObject::localTransform)
		.def_readwrite("localTranslation", &dyno::ModelObject::localTranslation)
		.def_readwrite("localRotation", &dyno::ModelObject::localRotation)
		.def_readwrite("localScale", &dyno::ModelObject::localScale)
		.def_readwrite("preRotation", &dyno::ModelObject::preRotation)
		.def_readwrite("pivot", &dyno::ModelObject::pivot)

		.def_readwrite("child", &dyno::ModelObject::child)
		.def_readwrite("parent", &dyno::ModelObject::parent)

		.def_readwrite("id", &dyno::ModelObject::id);

	py::class_<dyno::Bone, dyno::ModelObject, std::shared_ptr<dyno::Bone>>(m, "Bone")
		.def(py::init<>());

	py::class_<dyno::MeshInfo, dyno::ModelObject, std::shared_ptr<dyno::MeshInfo>>(m, "MeshInfo")
		.def(py::init<>())
		.def_readwrite("vertices", &dyno::MeshInfo::vertices)
		.def_readwrite("verticeId_pointId", &dyno::MeshInfo::verticeId_pointId)
		.def_readwrite("pointId_verticeId", &dyno::MeshInfo::pointId_verticeId)
		.def_readwrite("normals", &dyno::MeshInfo::normals)
		.def_readwrite("texcoords", &dyno::MeshInfo::texcoords)
		.def_readwrite("verticesColor", &dyno::MeshInfo::verticesColor)
		.def_readwrite("facegroup_triangles", &dyno::MeshInfo::facegroup_triangles)
		//.def_readwrite("facegroup_polygons", &dyno::MeshInfo::facegroup_polygons)

		.def_readwrite("materials", &dyno::MeshInfo::materials)

		.def_readwrite("boundingBox", &dyno::MeshInfo::boundingBox)
		.def_readwrite("boundingTransform", &dyno::MeshInfo::boundingTransform)

		.def_readwrite("boneIndices0", &dyno::MeshInfo::boneIndices0)
		.def_readwrite("boneWeights0", &dyno::MeshInfo::boneWeights0)
		.def_readwrite("boneIndices1", &dyno::MeshInfo::boneIndices1)
		.def_readwrite("boneWeights1", &dyno::MeshInfo::boneWeights1)
		.def_readwrite("boneIndices2", &dyno::MeshInfo::boneIndices2)
		.def_readwrite("boneWeights2", &dyno::MeshInfo::boneWeights2);

	py::class_<dyno::HierarchicalScene, dyno::Object, std::shared_ptr<dyno::HierarchicalScene>>(m, "HierarchicalScene")
		.def(py::init<>())
		.def("clear", &dyno::HierarchicalScene::clear)
		.def("find_mesh_index_by_name", &dyno::HierarchicalScene::findMeshIndexByName)
		.def("get_object_by_name", &dyno::HierarchicalScene::getObjectByName)
		.def("get_obj_index_by_name", &dyno::HierarchicalScene::getObjIndexByName)
		.def("get_bone_index_by_name", &dyno::HierarchicalScene::getBoneIndexByName)
		.def("update_bone_world_matrix", &dyno::HierarchicalScene::updateBoneWorldMatrix)
		.def("update_mesh_world_matrix", &dyno::HierarchicalScene::updateMeshWorldMatrix)
		.def("update_inverse_bind_matrix", &dyno::HierarchicalScene::updateInverseBindMatrix)
		.def("update_world_transform_by_key_frame", &dyno::HierarchicalScene::updateWorldTransformByKeyFrame)
		.def("get_vector_data_by_time", &dyno::HierarchicalScene::getVectorDataByTime)
		.def("find_max_smaller_index", &dyno::HierarchicalScene::findMaxSmallerIndex)
		.def("getBones", &dyno::HierarchicalScene::getBones)
		.def("skinAnimation", &dyno::HierarchicalScene::skinAnimation)
		.def("skinVerticesAnimation", &dyno::HierarchicalScene::skinVerticesAnimation)
		//.def("c_skinVerticesAnimation", &dyno::HierarchicalScene::c_skinVerticesAnimation)
		.def("getVerticesNormalInBindPose", &dyno::HierarchicalScene::getVerticesNormalInBindPose)

		.def("updatePoint2Vertice", &dyno::HierarchicalScene::updatePoint2Vertice)
		.def("UpdateJointData", &dyno::HierarchicalScene::UpdateJointData)
		.def("coutBoneHierarchial", &dyno::HierarchicalScene::coutBoneHierarchial)
		.def("updateSkinData", &dyno::HierarchicalScene::updateSkinData)
		.def("createLocalTransform", &dyno::HierarchicalScene::createLocalTransform)

		.def("coutMatrix", &dyno::HierarchicalScene::coutMatrix)
		.def("showJointInfo", &dyno::HierarchicalScene::showJointInfo)
		//.def("textureMeshTransform", &dyno::HierarchicalScene::textureMeshTransform)
		//.def("shapeTransform", &dyno::HierarchicalScene::shapeTransform)
		//.def("shapeToCenter", &dyno::HierarchicalScene::shapeToCenter)

		.def("getMeshes", &dyno::HierarchicalScene::getMeshes)
		.def("getObjectWorldMatrix", &dyno::HierarchicalScene::getObjectWorldMatrix)
		.def("computeTexMeshVerticesNormal", &dyno::HierarchicalScene::computeTexMeshVerticesNormal)
		.def("flipNormal", &dyno::HierarchicalScene::flipNormal)
		.def("getJointAnimation", &dyno::HierarchicalScene::getJointAnimation)

		.def_readwrite("mModelObjects", &dyno::HierarchicalScene::mModelObjects)
		.def_readwrite("mMeshes", &dyno::HierarchicalScene::mMeshes)
		.def_readwrite("mBones", &dyno::HierarchicalScene::mBones)
		.def_readwrite("mBoneRotations", &dyno::HierarchicalScene::mBoneRotations)
		.def_readwrite("mBoneTranslations", &dyno::HierarchicalScene::mBoneTranslations)
		.def_readwrite("mBoneScales", &dyno::HierarchicalScene::mBoneScales)
		.def_readwrite("mBoneWorldMatrix", &dyno::HierarchicalScene::mBoneWorldMatrix)
		.def_readwrite("mBoneInverseBindMatrix", &dyno::HierarchicalScene::mBoneInverseBindMatrix)

		.def_readwrite("mTimeStart", &dyno::HierarchicalScene::mTimeStart)
		.def_readwrite("mTimeEnd", &dyno::HierarchicalScene::mTimeEnd);
}