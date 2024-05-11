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

	declare_pointset<dyno::DataType3f>(m, "3f");
	declare_edgeSet<dyno::DataType3f>(m, "3f");
	declare_triangleSet<dyno::DataType3f>(m, "3f");
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

	declare_texture_mesh(m);
	declare_attribute(m);

	declare_point_set_to_triangle_set<dyno::DataType3f>(m, "3f");
}