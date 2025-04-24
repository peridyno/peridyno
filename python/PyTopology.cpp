#include "PyTopology.h"

#include "Topology/TextureMesh.h"
void declare_texture_mesh(py::module& m) {
	using Class = dyno::TextureMesh;
	using Parent = dyno::TopologyModule;
	std::string pyclass_name = std::string("TextureMesh");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
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
		.value("KINEMATIC_POSITIVE", Class::KinematicType::KINEMATIC_POSITIVE);

	py::enum_<typename Class::MaterialType>(Attribute, "MaterialType")
		.value("MATERIAL_MASK", Class::MaterialType::MATERIAL_MASK)
		.value("MATERIAL_FLUID", Class::MaterialType::MATERIAL_FLUID)
		.value("MATERIAL_RIGID", Class::MaterialType::MATERIAL_RIGID)
		.value("MATERIAL_ELASTIC", Class::MaterialType::MATERIAL_ELASTIC)
		.value("MATERIAL_PLASTIC", Class::MaterialType::MATERIAL_PLASTIC);

	py::enum_<typename Class::ObjectID>(Attribute, "ObjectID")
		.value("OBJECTID_MASK", Class::ObjectID::OBJECTID_MASK)
		.value("OBJECTID_INVALID", Class::ObjectID::OBJECTID_INVALID);
}


void pybind_topology(py::module& m)
{

	declare_point_set<dyno::DataType3f>(m, "3f");
	declare_edge_set<dyno::DataType3f>(m, "3f");
	declare_triangle_set<dyno::DataType3f>(m, "3f");
	declare_calculate_norm<dyno::DataType3f>(m, "3f");
	declare_height_field_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_discrete_elements_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_merge_triangle_set<dyno::DataType3f>(m, "3f");

	declare_neighbor_element_query<dyno::DataType3f>(m, "3f");
	declare_contacts_to_edge_set<dyno::DataType3f>(m, "3f");
	declare_contacts_to_point_set<dyno::DataType3f>(m, "3f");
	declare_neighbor_point_query<dyno::DataType3f>(m, "3f");
	declare_neighbor_triangle_query<dyno::DataType3f>(m, "3f");

	declare_calculate_bounding_box<dyno::DataType3f>(m, "3f");
	declare_collision_detection_broad_phase<dyno::DataType3f>(m, "3f");
	declare_collistion_detection_bounding_box<dyno::DataType3f>(m, "3f");
	declare_collistion_detection_triangle_set<dyno::DataType3f>(m, "3f");

	py::class_<dyno::PdActor, std::shared_ptr<dyno::PdActor>>(m, "PdActor");

	declare_joint<float>(m, "f");
	declare_ball_and_socket_joint<float>(m, "f");
	declare_slider_joint<float>(m, "f");
	declare_hinge_joint<float>(m, "f");
	declare_fixed_joint<float>(m, "f");
	declare_point_joint<float>(m, "f");

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

	py::enum_<typename dyno::ElementType>(m, "ElementType")
		.value("ET_BOX", dyno::ElementType::ET_BOX)
		.value("ET_TET", dyno::ElementType::ET_TET)
		.value("ET_CAPSULE", dyno::ElementType::ET_CAPSULE)
		.value("ET_SPHERE", dyno::ElementType::ET_SPHERE)
		.value("ET_TRI", dyno::ElementType::ET_TRI)
		.value("ET_Other", dyno::ElementType::ET_Other);

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
		//.def("is_contained_in", &dyno::OctreeNode::isContainedIn)
		.def("is_contained_strictly_in", &dyno::OctreeNode::isContainedStrictlyIn)
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
		.def("is_empty", &dyno::OctreeNode::isEmpty);

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
		.def(py::init<>());


	declare_texture_mesh(m);
	declare_attribute(m);
	declare_quad_set<dyno::DataType3f>(m, "3f");
	declare_point_set_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_anchor_point_to_point_set<dyno::DataType3f>(m, "3f");
	declare_bounding_box_to_edge_set<dyno::DataType3f>(m, "3f");
	declare_extract_edge_set_from_polygon_set<dyno::DataType3f>(m, "3f");
	declare_extract_triangle_set_from_polygon_set<dyno::DataType3f>(m, "3f");
	declare_extract_qaud_set_from_polygon_set<dyno::DataType3f>(m, "3f");
	declare_frame_to_point_set<dyno::DataType3f>(m, "3f");
	declare_marching_cubes<dyno::DataType3f>(m, "3f");
	declare_marching_cubes_helper<dyno::DataType3f>(m, "3f");
	declare_merge_simplex_set<dyno::DataType3f>(m, "3f");
	declare_point_set_to_point_set<dyno::DataType3f>(m, "3f");
	declare_quad_set_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_split_simplex_set<dyno::DataType3f>(m, "3f");
	declare_tetrahedron_set_to_point_set<dyno::DataType3f>(m, "3f");
	declare_texture_mesh_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_volume_clipper<dyno::DataType3f>(m, "3f");
	declare_calculate_maximum<dyno::DataType3f>(m, "3f");
	declare_calculate_minimum<dyno::DataType3f>(m, "3f");
	declare_animation_curve<dyno::DataType3f>(m, "3f");
	declare_discrete_elements<dyno::DataType3f>(m, "3f");
	declare_distance_field3D<dyno::DataType3f>(m, "3f");
	declare_frame<dyno::DataType3f>(m, "3f");
	declare_grid_hash<dyno::DataType3f>(m, "3f");
	declare_grid_set<dyno::DataType3f>(m, "3f");
	declare_height_field<dyno::DataType3f>(m, "3f");
	declare_hexahedron_set<dyno::DataType3f>(m, "3f");
	declare_linear_bvh<dyno::DataType3f>(m, "3f");
	declare_polygon_set<dyno::DataType3f>(m, "3f");


	declare_signed_distance_fieldt<dyno::DataType3f>(m, "3f");
	declare_simplex_set<dyno::DataType3f>(m, "3f");
	declare_sparse_grid_hash<dyno::DataType3f>(m, "3f");
	declare_sparse_octree<dyno::DataType3f>(m, "3f");
	declare_structured_point_set<dyno::DataType3f>(m, "3f");
	declare_tetrahedron_set<dyno::DataType3f>(m, "3f");
	declare_uniform_grid3D<dyno::DataType3f>(m, "3f");
	declare_unstructured_point_set<dyno::DataType3f>(m, "3f");
}