#include "PVTKPointSetSource.h"

#include "Utility/Function1Pt.h"

#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkPointSet.h"
#include "vtkPoints.h"

vtkStandardNewMacro(PVTKPointSetSource);

//----------------------------------------------------------------------------
PVTKPointSetSource::PVTKPointSetSource()
{
	this->SetNumberOfInputPorts(0);
	this->SetNumberOfOutputPorts(1);
}

//----------------------------------------------------------------------------
PVTKPointSetSource::~PVTKPointSetSource() = default;

int PVTKPointSetSource::RequestData(
	vtkInformation* vtkNotUsed(request),
	vtkInformationVector** vtkNotUsed(inputVector),
	vtkInformationVector* outputVector)
{
	printf("RequesData\n");
	if (m_point_set == nullptr)
	{
		printf("RequesData2\n");
		return 0;
	}

	// get the info object
	vtkInformation *outInfo = outputVector->GetInformationObject(0);

	// get the output
	vtkPointSet *output = vtkPointSet::SafeDownCast(
		outInfo->Get(vtkDataObject::DATA_OBJECT()));

	vtkPoints* pts = vtkPoints::New();

	auto device_pts = m_point_set->getPoints();

	int num_of_points = device_pts.size();
	printf("RD %d\n", num_of_points);

	dyno::CArray<dyno::Vector3f> host_pts;
	host_pts.resize(num_of_points);

	printf("HostPtsSize %d\n", host_pts.size());
	dyno::Function1Pt::copy(host_pts, device_pts);

	printf("Host Copy Finished\n");

	pts->Allocate(num_of_points);

	printf("Allocate Finished\n");
	
	for(int i = 0; i < num_of_points; i++)
	{
		//if (num_of_points > 2000)
			//printf("%.3lf %.3lf %.3lf\n", host_pts[i][0], host_pts[i][1], host_pts[i][2]);
		pts->InsertPoint(i, host_pts[i][0], host_pts[i][1], host_pts[i][2]);
	}

	pts->Squeeze();
	output->SetPoints(pts);
	pts->Delete();

	host_pts.release();

	return 1;
}


