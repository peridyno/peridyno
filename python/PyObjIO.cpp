#include "PyObjIO.h"

void pybind_objIO(py::module& m)
{
	declare_Obj_exporter<dyno::DataType3f>(m, "3f");
	declare_Obj_loader<dyno::DataType3f>(m, "3f");
	declare_Obj_point<dyno::DataType3f>(m, "3f");
	declare_PlyExporter<dyno::DataType3f>(m, "3f");
}