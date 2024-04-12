#include "PyModeling.h"
#include "CubeModel.h"
#include "Node/ParametricModel.h"

template <typename TDataType>
void declare_cube_model(py::module& m, std::string typestr) {
	using Class = dyno::CubeModel<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("CubeModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

void pybind_modeling(py::module& m) {
	declare_cube_model <dyno::DataType3f>(m, "3f");
}