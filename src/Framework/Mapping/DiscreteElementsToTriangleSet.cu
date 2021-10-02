#include "DiscreteElementsToTriangleSet.h"
#define NUM_POINT_SPHERE 49
#define NUM_TRIANGLE_SPHERE 72

namespace dyno
{
	typedef typename TOrientedBox3D<Real> Box3D;

	template<typename TDataType>
	DiscreteElementsToTriangleSet<TDataType>::DiscreteElementsToTriangleSet()
		: TopologyMapping()
	{

	}

	template<typename Triangle>
	__global__ void SetupTriangles(
		DArray<Vec3f> vertices,
		DArray<Triangle> indices,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Capsule3D> caps,
		DArray<Triangle3D> tris,
		ElementOffset elementOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boxes.size() + spheres.size() + tets.size() + caps.size() + tris.size()) return;
		ElementType eleType = checkElementType(tId, elementOffset);
		switch (eleType)
		{
			case CT_SPHERE:
			{
				int x_segments = 6;
				int y_segments = 6;
				int idx = tId;
				int offset_p = idx * NUM_POINT_SPHERE;

				Sphere3D sphere = spheres[idx];

				int summ = 0;
				for (int y = 0; y <= y_segments; y++)
				{
					for (int x = 0; x <= x_segments; x++)
					{
						Real xSegment = (Real)x / (Real)x_segments;
						Real ySegment = (Real)y / (Real)y_segments;
						Real xPos = cos(xSegment * 2.0f * M_PI) * sin(ySegment * M_PI) * sphere.radius + sphere.center[0];
						Real yPos = cos(ySegment * M_PI) * sphere.radius + sphere.center[1];
						Real zPos = sin(xSegment * 2.0f * M_PI) * sin(ySegment * M_PI) * sphere.radius + sphere.center[2];

						vertices[offset_p + summ] = Vec3f(xPos, yPos, zPos);
						summ++;
					}
				}

				//printf("Summ = % d %d %d\n", summ, vertices.size(), indices.size());
				summ = 0;
				int offset_t = idx * NUM_TRIANGLE_SPHERE;
				for (int i = 0; i < y_segments; i++)
				{
					for (int j = 0; j < x_segments; j++)
					{

						indices[offset_t + summ] = 
							Triangle(
								offset_p + i * (x_segments + 1) + j, 
								offset_p + (i + 1) * (x_segments + 1) + j,
								offset_p + (i + 1) * (x_segments + 1) + j + 1);
						summ ++;
						indices[offset_t + summ] = 
							Triangle(
								offset_p + i * (x_segments + 1) + j,
								offset_p + (i + 1) * (x_segments + 1) + j + 1,
								offset_p + i * (x_segments + 1) + j + 1);
						summ ++;
					}
				}
				//printf("Summ2 = % d\n", summ);
				break;
			}
			case CT_BOX:
			{

				int idx = tId - elementOffset.boxOffset;
				Box3D box = boxes[idx];

				Vec3f hx = box.u * box.extent[0];
				Vec3f hy = box.v * box.extent[1];
				Vec3f hz = box.w * box.extent[2];

				Vec3f hyz = hy + hz;
				Vec3f hxy = hx + hy;
				Vec3f hxz = hx + hz;

				Vec3f c = box.center;

				Vec3f v0 = c - hx - hyz;
				Vec3f v1 = c + hx - hyz;
				Vec3f v2 = c + hxz - hy;
				Vec3f v3 = c - hxy + hz;

				Vec3f v4 = c - hxz + hy;
				Vec3f v5 = c + hxy - hz;
				Vec3f v6 = c + hx + hyz;
				Vec3f v7 = c - hx + hyz;

				uint offset_point = spheres.size() * NUM_POINT_SPHERE;

				vertices[offset_point + idx * 8] = v0;
				vertices[offset_point + idx * 8 + 1] = v1;
				vertices[offset_point + idx * 8 + 2] = v2;
				vertices[offset_point + idx * 8 + 3] = v3;
				vertices[offset_point + idx * 8 + 4] = v4;
				vertices[offset_point + idx * 8 + 5] = v5;
				vertices[offset_point + idx * 8 + 6] = v6;
				vertices[offset_point + idx * 8 + 7] = v7;

				uint offset = idx * 8 + offset_point;
				uint offset_triangles = spheres.size() * NUM_TRIANGLE_SPHERE;

				indices[offset_triangles + idx * 12] = Triangle(offset + 0, offset + 1, offset + 2);
				indices[offset_triangles + idx * 12 + 1] = Triangle(offset + 0, offset + 2, offset + 3);

				indices[offset_triangles + idx * 12 + 2] = Triangle(offset + 0, offset + 4, offset + 5);
				indices[offset_triangles + idx * 12 + 3] = Triangle(offset + 0, offset + 5, offset + 1);

				indices[offset_triangles + idx * 12 + 4] = Triangle(offset + 4, offset + 7, offset + 6);
				indices[offset_triangles + idx * 12 + 5] = Triangle(offset + 4, offset + 6, offset + 5);

				indices[offset_triangles + idx * 12 + 6] = Triangle(offset + 1, offset + 5, offset + 6);
				indices[offset_triangles + idx * 12 + 7] = Triangle(offset + 1, offset + 6, offset + 2);

				indices[offset_triangles + idx * 12 + 8] = Triangle(offset + 2, offset + 6, offset + 7);
				indices[offset_triangles + idx * 12 + 9] = Triangle(offset + 2, offset + 7, offset + 3);

				indices[offset_triangles + idx * 12 + 10] = Triangle(offset + 0, offset + 3, offset + 7);
				indices[offset_triangles + idx * 12 + 11] = Triangle(offset + 0, offset + 7, offset + 4);

				break;
			}
			case CT_TET:
			{
				int idx = tId - elementOffset.tetOffset;
				Tet3D tet = tets[idx];

				Vec3f v0 = tet.v[0];
				Vec3f v1 = tet.v[1];
				Vec3f v2 = tet.v[2];
				Vec3f v3 = tet.v[3];


				uint offset_point = spheres.size() * NUM_POINT_SPHERE + 8 * boxes.size();

				vertices[offset_point + idx * 4] = v0;
				vertices[offset_point + idx * 4 + 1] = v1;
				vertices[offset_point + idx * 4 + 2] = v2;
				vertices[offset_point + idx * 4 + 3] = v3;
				

				uint offset = idx * 4 + offset_point;
				uint offset_triangles = spheres.size() * NUM_TRIANGLE_SPHERE + boxes.size() * 12;

				indices[offset_triangles + idx * 4] = Triangle(offset + 0, offset + 1, offset + 2);
				indices[offset_triangles + idx * 4 + 1] = Triangle(offset + 0, offset + 1, offset + 3);
				indices[offset_triangles + idx * 4 + 2] = Triangle(offset + 1, offset + 2, offset + 3);
				indices[offset_triangles + idx * 4 + 3] = Triangle(offset + 0, offset + 2, offset + 3);

				

				break;
			}
			case CT_SEG:
			{
				
				break;
			}
			case CT_TRI:
			{
				
				break;
			}
			default:
				break;
		}
	}

	template<typename TDataType>
	bool DiscreteElementsToTriangleSet<TDataType>::apply()
	{
		if (this->outTriangleSet()->isEmpty())
		{
			this->outTriangleSet()->allocate();
		}

		auto inTopo = this->inDiscreteElements()->getDataPtr();

		//printf("====================================================== inside box update\n");
		auto& spheres = inTopo->getSpheres();
		auto& boxes = inTopo->getBoxes();
		auto& tets = inTopo->getTets();
		auto& caps = inTopo->getCaps();
		auto& tris = inTopo->getTris();
		ElementOffset elementOffset = inTopo->calculateElementOffset();

		int numOfSpheres = spheres.size();
		int numOfBoxes = boxes.size();
		int numOfTets = tets.size();

		printf("Num of tets = %d\n", numOfTets);

		auto triSet = this->outTriangleSet()->getDataPtr();

		auto& points = triSet->getPoints();
		auto indices = triSet->getTriangles();

		int numOfVertices = 8 * numOfBoxes + 4 * numOfTets + NUM_POINT_SPHERE * numOfSpheres;
		int numOfTriangles = 12 * numOfBoxes + 4 * numOfTets + NUM_TRIANGLE_SPHERE * numOfSpheres;

		points.resize(numOfVertices);
		indices->resize(numOfTriangles);

		cuExecute(numOfBoxes,
			SetupTriangles,
			points,
			*indices,
			boxes,
			spheres,
			tets,
			caps,
			tris,
			elementOffset);

		return true;
	}

	DEFINE_CLASS(DiscreteElementsToTriangleSet);
}