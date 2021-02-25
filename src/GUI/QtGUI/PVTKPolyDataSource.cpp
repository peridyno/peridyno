#include "PVTKPolyDataSource.h"

#include "Utility/Function1Pt.h"

#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkPoints.h"

vtkStandardNewMacro(PVTKPolyDataSource);

//----------------------------------------------------------------------------
PVTKPolyDataSource::PVTKPolyDataSource()
{
	this->SetNumberOfInputPorts(0);
	this->SetNumberOfOutputPorts(1);
}

//----------------------------------------------------------------------------
PVTKPolyDataSource::~PVTKPolyDataSource() = default;

int PVTKPolyDataSource::RequestData(
	vtkInformation* vtkNotUsed(request),
	vtkInformationVector** vtkNotUsed(inputVector),
	vtkInformationVector* outputVector)
{

	if (m_tri_set == nullptr)
	{
		return 0;
	}

	// get the info object
	vtkInformation *outInfo = outputVector->GetInformationObject(0);

	// get the output
	vtkPolyData *output = vtkPolyData::SafeDownCast(
		outInfo->Get(vtkDataObject::DATA_OBJECT()));

	vtkPoints* pts = vtkPoints::New();

	auto device_pts = m_tri_set->getPoints();

	int num_of_points = device_pts.size();
	dyno::HostArray<dyno::Vector3f> host_pts;
	host_pts.resize(num_of_points);
	dyno::Function1Pt::copy(host_pts, device_pts);

	pts->Allocate(num_of_points);

	for(int i = 0; i < num_of_points; i++)
	{
		pts->InsertPoint(i, host_pts[i][0], host_pts[i][1], host_pts[i][2]);
	}

	auto device_triangles = m_tri_set->getTriangles();
	int num_of_triangles = device_triangles->size();

	dyno::HostArray<dyno::TopologyModule::Triangle> host_triangles;
	host_triangles.resize(num_of_triangles);
	dyno::Function1Pt::copy(host_triangles, *device_triangles);


	vtkCellArray *polys;
	polys = vtkCellArray::New();
	polys->Allocate(num_of_triangles, 3);
	vtkIdType ids[3];
	for (int i = 0; i < num_of_triangles; i++)
	{
		ids[0] = host_triangles[i][0];
		ids[1] = host_triangles[i][1];
		ids[2] = host_triangles[i][2];
		polys->InsertNextCell(3, ids);
	}
	pts->Squeeze();
	output->SetPoints(pts);
	pts->Delete();

	polys->Squeeze();
	output->SetPolys(polys);
	polys->Delete();

	host_pts.release();
	host_triangles.release();

	return 1;
}

//----------------------------------------------------------------------------
int PVTKPolyDataSource::RequestInformation(
	vtkInformation *vtkNotUsed(request),
	vtkInformationVector **vtkNotUsed(inputVector),
	vtkInformationVector *outputVector)
{
	// get the info object
	vtkInformation *outInfo = outputVector->GetInformationObject(0);

	outInfo->Set(CAN_HANDLE_PIECE_REQUEST(), 1);

	return 1;
}

