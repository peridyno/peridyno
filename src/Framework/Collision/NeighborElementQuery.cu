#include "NeighborElementQuery.h"
#include "CollisionDetectionAlgorithm.h"

#include "Collision/CollisionDetectionBroadPhase.h"

#include "Topology/Primitive3D.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(NeighborElementQuery, TDataType)
		typedef typename TOrientedBox3D<Real> Box3D;


	template<typename TDataType>
	NeighborElementQuery<TDataType>::NeighborElementQuery()
		: ComputeModule()
	{
		this->inRadius()->setValue(Real(0.011));

		m_broadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
		//fout.open("data_Oct_without_arrange.txt");
	}

	template<typename TDataType>
	NeighborElementQuery<TDataType>::~NeighborElementQuery()
	{
	}

	template<typename Real, typename Coord>
	__global__ void NEQ_SetupAABB(
		DArray<AABB> boundingBox,
		DArray<Coord> position,
		Real radius)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		AABB box;
		Coord p = position[pId];
		box.v0 = p - radius;
		box.v1 = p + radius;

		boundingBox[pId] = box;
	}

	template<typename Box3D>
	__global__ void NEQ_SetupAABB(
		DArray<AABB> boundingBox,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Capsule3D> caps,
		DArray<Triangle3D> tris,
		ElementOffset elementOffset,
		Real boundary_expand)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		ElementType eleType = checkElementType(tId, elementOffset);

		//Real boundary_expand = 0.0075f;

		AABB box;
		switch (eleType)
		{
		case CT_SPHERE:
		{
			box = spheres[tId].aabb();

			break;
		}
		case CT_BOX:
		{
			box = boxes[tId - elementOffset.boxOffset].aabb();
			box.v0 -= boundary_expand;
			box.v1 += boundary_expand;
			break;
		}
		case CT_TET:
		{
			box = tets[tId - elementOffset.tetOffset].aabb();
			box.v0 -= boundary_expand;
			box.v1 += boundary_expand;
			break;
		}
		case CT_SEG:
		{
			box = caps[tId - elementOffset.segOffset].aabb();
			box.v0 -= boundary_expand;
			box.v1 += boundary_expand;
			break;
		}
		case CT_TRI:
		{
			box = tris[tId - elementOffset.triOffset].aabb();
			box.v0 -= boundary_expand;
			box.v1 += boundary_expand;
			box.v0 -= boundary_expand;
			box.v1 += boundary_expand;
			/*printf("%.3lf %.3lf %.3lf\n%.3lf %.3lf %.3lf\n=========\n",
				box.v0[0], box.v0[1], box.v0[2],
				box.v1[0], box.v1[1], box.v1[2]);*/
			break;
		}
		default:
			break;
		}

		boundingBox[tId] = box;
	}

	


	template<typename Box3D>
	__global__ void NEQ_Narrow_Count(
		DArrayList<int> nbr,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Real> tets_sdf,
		DArray<int> tet_body_ids,
		DArray<TopologyModule::Tetrahedron> tet_element_ids,
		DArray<Capsule3D> caps,
		DArray<Triangle3D> triangles,
		DArray<int> count,
		ElementOffset elementOffset,
		NbrFilter filter,
		Real boundary_expand)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nbr.size()) return;

		ElementType eleType_i = checkElementType(tId, elementOffset);

		int start_box = elementOffset.boxOffset;
		int start_tet = elementOffset.tetOffset;
		//Real boundary_expand = 0.0;
		int cnt = 0;

		switch (eleType_i)
		{
		case CT_SPHERE:
		{
			//break;
			//int nbSize = nbr.getNeighborSize(tId);
			//if(nbSize != 0)
				//printf("nbSize = %d\n", nbSize);
			List<int>& list_i = nbr[tId];
			int nbSize = list_i.size();

			for (int ne = 0; ne < nbSize; ne++)
			{
				//int j = nbr.getElement(tId, ne);
				int j = list_i[ne];

				ElementType eleType_j = checkElementType(j, elementOffset);

				//printf("%d %d\n", eleType_i, eleType_j);

				switch (eleType_j)
				{
				case CT_SPHERE:
				{

					Real proj_dist = (spheres[j].center - spheres[tId].center).norm();
					if (filter.sphere_sphere && proj_dist < spheres[tId].radius + spheres[j].radius)
					{
						cnt++;
					}
					break;
				}
				case CT_BOX:
				{
					Point3D pos_sphere(spheres[tId].center);
					Coord3D proj_pos = pos_sphere.project(boxes[j - elementOffset.boxOffset]).origin;

					if (filter.sphere_box)
						if ((proj_pos - spheres[tId].center).norm() < spheres[tId].radius + boundary_expand
							|| pos_sphere.inside(boxes[j - elementOffset.boxOffset]))
						{
							cnt++;

						}
					break;
				}
				case CT_TET:
				{
					//printf("@@@@@@@@@@@@@@@@@@ %d\n", tId);
					if (filter.sphere_tet && filter.tet_sdf && tets_sdf.size() > 0)
					{
						Point3D pos_sphere(spheres[tId].center);
						int idx;
						Bool tmp;
						Coord3D proj_pos = pos_sphere.project(tets[j - elementOffset.tetOffset], tmp, &idx).origin;

						int id1 = (idx + 1) % 4;
						int id2 = (idx + 2) % 4;
						int id3 = (idx + 3) % 4;

						Triangle3D t3d_tmp = Triangle3D(
							tets[j - elementOffset.tetOffset].v[id1],
							tets[j - elementOffset.tetOffset].v[id2],
							tets[j - elementOffset.tetOffset].v[id3]
						);

						if (tets_sdf[4 * (j - elementOffset.tetOffset) + id1] < EPSILON &&
							tets_sdf[4 * (j - elementOffset.tetOffset) + id2] < EPSILON &&
							tets_sdf[4 * (j - elementOffset.tetOffset) + id3] < EPSILON)
						{
							if (((proj_pos - spheres[tId].center).norm() < spheres[tId].radius + boundary_expand
								&& (((proj_pos - spheres[tId].center) / (proj_pos - spheres[tId].center).norm()).cross(t3d_tmp.normal() / t3d_tmp.normal().norm())).norm() < EPSILON)
								|| pos_sphere.inside(tets[j - elementOffset.tetOffset]))
							{
								cnt++;
							}
						}
						else if (pos_sphere.inside(tets[j - elementOffset.tetOffset]))
						{
							cnt++;
						}
					}
					else if (filter.sphere_tet)
					{
						Point3D pos_sphere(spheres[tId].center);
						int idx;
						Bool tmp;
						Coord3D proj_pos = pos_sphere.project(tets[j - elementOffset.tetOffset], tmp, &idx).origin;
						if ((proj_pos - spheres[tId].center).norm() < spheres[tId].radius + boundary_expand
							|| pos_sphere.inside(tets[j - elementOffset.tetOffset]))
						{
							cnt++;
						}
					}
					break;
				}
				case CT_SEG:
				{
					Point3D pos_sphere(spheres[tId].center);
					Segment3D segment_tmp = caps[j - elementOffset.segOffset].segment;
					if (filter.sphere_capsule)
						if (pos_sphere.distance(segment_tmp) < spheres[tId].radius + caps[j - elementOffset.segOffset].radius + boundary_expand)
						{
							//printf("CNT_OKKKK\n");
							//if((j - elementOffset.segOffset) % 39 == 0 || (j - elementOffset.segOffset) % 39 == 38)
							cnt++;
							/*else if ((pos_sphere.project(segment_tmp).origin - pos_sphere.origin).dot(segment_tmp.direction()) < EPSILON)
							{
								cnt++;
							}*/
						}
					break;
				}
				case CT_TRI:
				{
					Point3D pos_sphere(spheres[tId].center);
					Triangle3D tri_tmp = triangles[j - elementOffset.triOffset];
					if (filter.sphere_tri)
						if (pos_sphere.distance(tri_tmp) < spheres[tId].radius + 1.75 * boundary_expand && pos_sphere.distance(tri_tmp) > EPSILON
							&& (((pos_sphere.project(tri_tmp).origin - pos_sphere.origin) / (pos_sphere.project(tri_tmp).origin - pos_sphere.origin).norm()).cross(tri_tmp.normal() / tri_tmp.normal().norm())).norm() < 0.001
							)
						{
							//printf("CNT_OKKKK\n");
							cnt++;
						}
					break;
				}
				default:
					break;
				}
			}
			break;

		}
		case CT_BOX:
		{
			//int nbSize = nbr.getNeighborSize(tId);
			List<int>& list_i = nbr[tId];
			int nbSize = list_i.size();

			for (int ne = 0; ne < nbSize; ne++)
			{
				//int j = nbr.getElement(tId, ne);
				int j = list_i[ne];

				ElementType eleType_j = checkElementType(j, elementOffset);

				switch (eleType_j)
				{
				case CT_SPHERE:
				{
					Point3D pos_sphere(spheres[j].center);
					Coord3D proj_pos = pos_sphere.project(boxes[tId - elementOffset.boxOffset]).origin;
					if (filter.sphere_box)
						if ((proj_pos - spheres[j].center).norm() < spheres[j].radius + boundary_expand
							|| pos_sphere.inside(boxes[tId - elementOffset.boxOffset]))
						{
							cnt++;
						}
					//printf("sphere!!!! %d\n", j);
					break;
				}
				case CT_BOX:
				{
					TManifold<Real> manifold;

					auto boxA = boxes[tId - elementOffset.boxOffset];
					auto boxB = boxes[j - elementOffset.boxOffset];
					CollisionDetection<Real>::request(manifold, boxA, boxB);

					cnt += manifold.contactCount;

// 					Coord3D inter_norm, p1, p2;
// 					Real inter_dist;
// 					if (filter.box_box)
// 					{
// 						if (boxes[tId - elementOffset.boxOffset].point_intersect(boxes[j - elementOffset.boxOffset], inter_norm, inter_dist, p1, p2))
// 						{
// 							cnt++;
// 						}
// 						else if (boxes[j - elementOffset.boxOffset].point_intersect(boxes[tId - elementOffset.boxOffset], inter_norm, inter_dist, p1, p2))
// 						{
// 							cnt++;
// 						}
// 					}
					break;
				}
				case CT_TET:
				{
					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if (filter.box_tet)
						if (boxes[tId - elementOffset.boxOffset].point_intersect(tets[j - elementOffset.tetOffset], inter_norm, inter_dist, p1, p2))
						{
							cnt++;
						}
					break;
				}
				case CT_SEG:
				{
					if (filter.box_capsule)
					{
						Segment3D segment_tmp = caps[j - elementOffset.segOffset].segment;
						Segment3D segment_intersect;
						Segment3D seg_prox = segment_tmp.proximity(boxes[tId - elementOffset.boxOffset]);
						if (segment_tmp.intersect(boxes[tId - elementOffset.boxOffset], segment_intersect))
						{
							cnt++;
						}
						else if (seg_prox.length() < caps[j - elementOffset.segOffset].radius)
						{
							//if(seg_prox.direction().dot(segment_tmp.direction()) < EPSILON)//////to delete
							cnt++;
						}
					}
					break;
				}
				case CT_TRI:
				{
					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if (filter.box_tri)
					{
						if (boxes[tId - elementOffset.boxOffset].point_intersect(triangles[j - elementOffset.triOffset], inter_norm, inter_dist, p1, p2))
						{
							cnt++;
						}
					}
					break;
				}
				default:
					break;
				}
			}
			break;
		}
		case CT_TET:
		{
			//printf("nbSize = %d   %d\n", tId, nbr.getNeighborSize(tId));

			//int nbSize = nbr.getNeighborSize(tId);
			//if(nbSize != 0)
			List<int>& list_i = nbr[tId];
			int nbSize = list_i.size();
			for (int ne = 0; ne < nbSize; ne++)
			{
				//if(nbr.getElementSize() <= nbr.getElementIndex(tId, ne))
				//	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
				//int j = nbr.getElement(tId, ne);
				int j = list_i[ne];

				ElementType eleType_j = checkElementType(j, elementOffset);
				
				//break;
				//if(false)
				switch (eleType_j)
				{
				case CT_SPHERE:
				{
					//Point3D pos_sphere(spheres[j].center);
					//Coord3D proj_pos = pos_sphere.project(tets[tId - elementOffset.tetOffset]).origin;
					if (filter.sphere_tet)
					{
						//printf("j idx = %d   %d\n", j, spheres.size());
						Point3D pos_sphere(spheres[j].center);
						int idx;
						Bool tmp;

						Coord3D proj_pos = pos_sphere.project(tets[tId - elementOffset.tetOffset], tmp, &idx).origin;

						if (filter.tet_sdf && tets_sdf.size() > 0)
						{
							int id1 = (idx + 1) % 4;
							int id2 = (idx + 2) % 4;
							int id3 = (idx + 3) % 4;

							Triangle3D t3d_tmp = Triangle3D(
								tets[tId - elementOffset.tetOffset].v[id1],
								tets[tId - elementOffset.tetOffset].v[id2],
								tets[tId - elementOffset.tetOffset].v[id3]
							);

							if (tets_sdf[4 * (tId - elementOffset.tetOffset) + id1] < EPSILON &&
								tets_sdf[4 * (tId - elementOffset.tetOffset) + id2] < EPSILON &&
								tets_sdf[4 * (tId - elementOffset.tetOffset) + id3] < EPSILON)
							{
								if (((proj_pos - spheres[j].center).norm() < spheres[j].radius + boundary_expand
									&& (((proj_pos - spheres[j].center) / (proj_pos - spheres[j].center).norm()).cross(t3d_tmp.normal() / t3d_tmp.normal().norm())).norm() < EPSILON)
									|| pos_sphere.inside(tets[tId - elementOffset.tetOffset]))
								{
									cnt++;
								}
							}
							else
							{
								if (pos_sphere.inside(tets[tId - elementOffset.tetOffset]))
								{
									cnt++;
								}
							}
						}
						else
						{
							if ((proj_pos - spheres[j].center).norm() < spheres[j].radius + boundary_expand
								|| pos_sphere.inside(tets[tId - elementOffset.tetOffset]))
							{
								cnt++;
							}
						}
					}
					break;
				}
				case CT_BOX:
				{
					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if (filter.box_tet)
						if (boxes[j - elementOffset.boxOffset].point_intersect(tets[tId - elementOffset.tetOffset], inter_norm, inter_dist, p1, p2))
						{
							cnt++;
						}
					break;
				}
				case CT_TET:
				{
					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if (filter.tet_tet)
					{

						bool tmp_tet = false;
						AABB interBox;
						if (!tets[tId - elementOffset.tetOffset].aabb().intersect(tets[j - elementOffset.tetOffset].aabb(), interBox))
						{
							break;
						}
						if (filter.tet_neighbor_filter
							&& tet_body_ids.size() > 0
							&& tet_body_ids[tId - elementOffset.tetOffset] == tet_body_ids[j - elementOffset.tetOffset])
						{
							bool tmp = false;
							for (int iii = 0; iii < 4; iii++)
							{
								for (int jjj = 0; jjj < 4; jjj++)
									if (tet_element_ids[tId - elementOffset.tetOffset][iii]
										== tet_element_ids[j - elementOffset.tetOffset][jjj])
									{
										tmp = true;
										break;
									}
								if (tmp)
									break;
							}
							if (tmp)
								break;

						}
						if (tets[tId - elementOffset.tetOffset].intersect(tets[j - elementOffset.tetOffset], inter_norm, inter_dist, p1, p2, false))
						{
							if (abs(inter_dist) > 3 * EPSILON)
							{
								cnt++;
								tmp_tet = true;
							}
						}
						if (tets[j - elementOffset.tetOffset].intersect(tets[tId - elementOffset.tetOffset], inter_norm, inter_dist, p1, p2, false))
						{
							if(abs(inter_dist) > 3 * EPSILON && tmp_tet == false)
							{ 
								cnt++;
							}
						}
						
					}
					break;
				}
				case CT_SEG:
				{
					if (filter.tet_capsule)
					{
						Segment3D segment_tmp = caps[j - elementOffset.segOffset].segment;
						Line3D line_tmp(segment_tmp.v0, segment_tmp.direction());
						Segment3D segment_tmp2 = segment_tmp.proximity(tets[tId - elementOffset.tetOffset]);
						Segment3D segment_intersect;
						bool intersect_1 = false;
						if (line_tmp.intersect(tets[tId - elementOffset.tetOffset], segment_intersect))
						{
							Real left = (segment_intersect.v0 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							Real right = (segment_intersect.v1 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							if (right < left)
							{
								Real tmp = left;
								left = right;
								right = tmp;
							}
							Real maxx = (segment_tmp.v1 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							if (right < 0 || left > maxx)
							{

							}
							else
							{
								intersect_1 = true;
								cnt++;
							}
						}
						if (!intersect_1)
						{
							if (segment_tmp2.length() < caps[j - elementOffset.segOffset].radius)
							{
								cnt++;
							}
						}
					}
					break;
				}
				case CT_TRI:
				{
					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if (filter.tet_tri)
					{
						if (tets[tId - elementOffset.tetOffset].intersect(triangles[j - elementOffset.triOffset], inter_norm, inter_dist, p1, p2))
						{
							cnt++;
						}
					}
					break;
				}
				default:
					break;
				}
			}
			break;
		}
		case CT_SEG:
		{
			//int nbSize = nbr.getNeighborSize(tId);
			List<int>& list_i = nbr[tId];
			int nbSize = list_i.size();
			for (int ne = 0; ne < nbSize; ne++)
			{
				//int j = nbr.getElement(tId, ne);
				int j = list_i[ne];

				ElementType eleType_j = checkElementType(j, elementOffset);

				switch (eleType_j)
				{

				case CT_SPHERE:
				{
					Point3D pos_sphere(spheres[j].center);
					Segment3D segment_tmp = caps[tId - elementOffset.segOffset].segment;
					if (filter.sphere_capsule)
						if (pos_sphere.distance(segment_tmp) < spheres[j].radius + caps[tId - elementOffset.segOffset].radius + boundary_expand)
						{
							//printf("CNT_OKKKK\n");
							//cnt++;
							//printf("CNT_OKKKK\n");
							//if ((tId - elementOffset.segOffset) % 39 == 0 || (tId - elementOffset.segOffset) % 39 == 38)
							cnt++;
							/*else if ((pos_sphere.project(segment_tmp).origin - pos_sphere.origin).dot(segment_tmp.direction()) < EPSILON)
							{
								cnt++;
							}*/
						}
					break;
				}
				case CT_BOX:
				{
					if (filter.box_capsule)
					{
						Segment3D segment_tmp = caps[tId - elementOffset.segOffset].segment;
						Segment3D segment_intersect;
						Segment3D seg_prox = segment_tmp.proximity(boxes[j - elementOffset.boxOffset]);
						if (segment_tmp.intersect(boxes[j - elementOffset.boxOffset], segment_intersect))
						{
							cnt++;
						}
						else if (seg_prox.length() < caps[tId - elementOffset.segOffset].radius)
						{
							//if (seg_prox.direction().dot(segment_tmp.direction()) < EPSILON)//////to delete
							cnt++;
						}
					}
					break;
				}
				case CT_TET:
				{
					if (filter.tet_capsule)
					{
						Segment3D segment_tmp = caps[tId - elementOffset.segOffset].segment;
						Line3D line_tmp(segment_tmp.v0, segment_tmp.direction());
						Segment3D segment_tmp2 = segment_tmp.proximity(tets[j - elementOffset.tetOffset]);
						Segment3D segment_intersect;
						bool intersect_1 = false;
						if (line_tmp.intersect(tets[j - elementOffset.tetOffset], segment_intersect))
						{
							Real left = (segment_intersect.v0 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							Real right = (segment_intersect.v1 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							if (right < left)
							{
								Real tmp = left;
								left = right;
								right = tmp;
							}
							Real maxx = (segment_tmp.v1 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							if (right < 0 || left > maxx)
							{

							}
							else
							{
								intersect_1 = true;
								cnt++;
							}
						}
						if (!intersect_1)
						{
							if (segment_tmp2.length() < caps[tId - elementOffset.segOffset].radius)
							{
								cnt++;
							}
						}
					}
					break;
				}
				case CT_SEG:
				{
					if (filter.capsule_capsule)
					{
						Segment3D segment_1 = caps[j - elementOffset.segOffset].segment;
						Segment3D segment_2 = caps[tId - elementOffset.segOffset].segment;
						if ((segment_1.proximity(segment_2)).length() < caps[j - elementOffset.segOffset].radius + caps[tId - elementOffset.segOffset].radius)
						{
							if (abs(j - tId) >= 3)
								cnt++;
						}
					}
					break;
				}
				case CT_TRI:
				{
					if (filter.capsule_tri)
					{
						//Segment3D segment_1 = caps[j - elementOffset.segOffset].segment;
						Segment3D segment = caps[tId - elementOffset.segOffset].segment;
						Triangle3D triangle = triangles[j - elementOffset.triOffset];
						Point3D p3d;
						if (segment.intersect(triangle, p3d))
						{
							////if (abs(j - tId) >= 3)
							cnt++;
						}
						else if ((segment.proximity(triangle)).length() < caps[tId - elementOffset.segOffset].radius)
						{
							cnt++;
						}
					}
					break;
				}
				default:
					break;
				}
			}
			break;
		}
		case CT_TRI:
		{
			//int nbSize = nbr.getNeighborSize(tId);
			List<int>& list_i = nbr[tId];
			int nbSize = list_i.size();

			for (int ne = 0; ne < nbSize; ne++)
			{
				//int j = nbr.getElement(tId, ne);
				int j = list_i[ne];

				ElementType eleType_j = checkElementType(j, elementOffset);

				switch (eleType_j)
				{

				case CT_SPHERE:
				{
					Point3D pos_sphere(spheres[j].center);
					Triangle3D tri_tmp = triangles[tId - elementOffset.triOffset];
					if (filter.sphere_tri)
						if (pos_sphere.distance(tri_tmp) < spheres[j].radius + 1.75 * boundary_expand && pos_sphere.distance(tri_tmp) > EPSILON
							//&& ((pos_sphere.project(tri_tmp).origin - pos_sphere.origin) / (pos_sphere.project(tri_tmp).origin - pos_sphere.origin).norm()).dot(tri_tmp.normal() / tri_tmp.normal().norm()) < EPSILON
							&& (((pos_sphere.project(tri_tmp).origin - pos_sphere.origin) / (pos_sphere.project(tri_tmp).origin - pos_sphere.origin).norm()).cross(tri_tmp.normal() / tri_tmp.normal().norm())).norm() < 0.001
							)
						{
							//printf("CNT_OKKKK\n");
							cnt++;
						}
					break;
				}
				case CT_BOX:
				{
					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if (filter.box_tri)
					{
						if (boxes[j - elementOffset.boxOffset].point_intersect(triangles[tId - elementOffset.triOffset], inter_norm, inter_dist, p1, p2))
						{
							cnt++;
						}
					}
					break;
				}
				case CT_TET:
				{

					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if (filter.tet_tri)
					{
						if (tets[j - elementOffset.tetOffset].intersect(triangles[tId - elementOffset.triOffset], inter_norm, inter_dist, p1, p2))
						{
							cnt++;
						}
					}
					break;
				}
				case CT_SEG:
				{
					if (filter.capsule_tri)
					{
						//Segment3D segment_1 = caps[j - elementOffset.segOffset].segment;
						Segment3D segment = caps[j - elementOffset.segOffset].segment;
						Triangle3D triangle = triangles[tId - elementOffset.triOffset];
						Point3D p3d;
						if (segment.intersect(triangle, p3d))
						{
							cnt++;
						}
						else if ((segment.proximity(triangle)).length() < caps[j - elementOffset.segOffset].radius)
						{
							cnt++;
						}
					}
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
			break;
		}
		default:
			break;
		}

		count[tId] = cnt;
	}

	template<typename Box3D, typename NeighborConstraints>
	__global__ void NEQ_Narrow_Set(
		DArrayList<int> nbr,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Real> tets_sdf,
		DArray<int> tet_body_ids,
		DArray<TopologyModule::Tetrahedron> tet_element_ids,
		DArray<Capsule3D> caps,
		DArray<Triangle3D> tris,
		DArray<NeighborConstraints> nbr_cons,
		DArray<int> prefix,
		ElementOffset elementOffset,
		NbrFilter filter,
		Real boundary_expand)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nbr.size()) return;
		int cnt = 0;

		ElementType eleType_i = checkElementType(tId, elementOffset);
		//printf("box offset: %d\n", elementOffset.boxOffset);
		switch (eleType_i)
		{
		case CT_SPHERE:
		{
			//break;
			//int nbSize = nbr.getNeighborSize(tId);
			List<int>& list_i = nbr[tId];
			int nbSize = list_i.size();
			for (int ne = 0; ne < nbSize; ne++)
			{
				int j = list_i[ne];
				ElementType eleType_j = checkElementType(j, elementOffset);
				switch (eleType_j)
				{
				case CT_SPHERE:
				{
					Real proj_dist = (spheres[j].center - spheres[tId].center).norm();
					if (filter.sphere_sphere)
						if (proj_dist < spheres[tId].radius + spheres[j].radius)
						{

							Coord3D inter_norm = (spheres[j].center - spheres[tId].center) / proj_dist;
							Coord3D p1 = spheres[j].center - inter_norm * spheres[j].radius;
							Coord3D p2 = spheres[tId].center + inter_norm * spheres[tId].radius;
							Real inter_dist = spheres[tId].radius + spheres[j].radius - proj_dist;

							//nbr_out.setElement(tId, cnt, j);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;//nbr_out.getElementIndex(tId, cnt);
							nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_FLUID_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = inter_dist;
							cnt++;
							//printf("?????????????\n");

						}
					break;
				}
				case CT_BOX:
				{
					if (filter.sphere_box)
					{
						Point3D pos_sphere(spheres[tId].center);
						Coord3D proj_pos = pos_sphere.project(boxes[j - elementOffset.boxOffset]).origin;
						if (pos_sphere.inside(boxes[j - elementOffset.boxOffset]))
						{
							Coord3D inter_norm = -(proj_pos - spheres[tId].center) / (proj_pos - spheres[tId].center).norm();
							Coord3D p1 = proj_pos;
							Coord3D p2 = spheres[tId].center - inter_norm * spheres[tId].radius;
							Real inter_dist = spheres[tId].radius + (proj_pos - spheres[tId].center).norm();

							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;//nbr_out.getElementIndex(tId, cnt);
							nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
							//nbr_cons[idx_con] = NeighborConstraints(j, tId, 8, p1, p2, 0.0f, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = inter_dist;
							cnt++;
						}
						else if ((proj_pos - spheres[tId].center).norm() < spheres[tId].radius + boundary_expand)
						{
							Coord3D inter_norm = (proj_pos - spheres[tId].center) / (proj_pos - spheres[tId].center).norm();
							Coord3D p1 = proj_pos;
							Coord3D p2 = spheres[tId].center + inter_norm * spheres[tId].radius;
							Real inter_dist = spheres[tId].radius - (proj_pos - spheres[tId].center).norm();

							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;//nbr_out.getElementIndex(tId, cnt);
							nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_FLUID_STICKINESS, p1, p2, inter_norm, Coord3D(0, 0, 0));
							//nbr_cons[idx_con] = NeighborConstraints(j, tId, 6, p1, p2, 0.0f, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = 0.0f;//inter_dist;
							cnt++;
							//printf("sphere!! %d\n", tId);
						}
					}
					break;
				}
				case CT_TET:
				{
					if (filter.sphere_tet)
					{
						Point3D pos_sphere(spheres[tId].center);
						int idx;
						Bool tmp;
						Coord3D proj_pos = pos_sphere.project(tets[j - elementOffset.tetOffset], tmp, &idx).origin;

						if (filter.tet_sdf)// && tets_sdf.size() > 0)
						{
							int id1 = (idx + 1) % 4;
							int id2 = (idx + 2) % 4;
							int id3 = (idx + 3) % 4;

							Triangle3D t3d_tmp = Triangle3D(
								tets[j - elementOffset.tetOffset].v[id1],
								tets[j - elementOffset.tetOffset].v[id2],
								tets[j - elementOffset.tetOffset].v[id3]
							);

							if (tets_sdf[4 * (j - elementOffset.tetOffset) + id1] < EPSILON &&
								tets_sdf[4 * (j - elementOffset.tetOffset) + id2] < EPSILON &&
								tets_sdf[4 * (j - elementOffset.tetOffset) + id3] < EPSILON)
							{
								//printf("========aaaa=aaaa======\n");
								if (pos_sphere.inside(tets[j - elementOffset.tetOffset]))
								{
									//printf("========bbbbb======\n");
									Coord3D inter_norm = -(proj_pos - spheres[tId].center) / (proj_pos - spheres[tId].center).norm();
									Coord3D p1 = proj_pos;
									Coord3D p2 = spheres[tId].center - inter_norm * spheres[tId].radius;
									Real inter_dist = spheres[tId].radius + (proj_pos - spheres[tId].center).norm();

									//nbr_out.setElement(tId, cnt, j);
									//int idx_con = nbr_out.getElementIndex(tId, cnt);
									//list_j.insert(j);
									int idx_con = prefix[tId] + cnt;
									nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_FLUID_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = inter_dist;
									cnt++;
								}
								else if ((proj_pos - spheres[tId].center).norm() < spheres[tId].radius + boundary_expand
									&& (((proj_pos - spheres[tId].center) / (proj_pos - spheres[tId].center).norm()).cross(t3d_tmp.normal() / t3d_tmp.normal().norm())).norm() < EPSILON)
								{
									//printf("YYEESS\n");
										//printf("========vvvvvv======\n");
									Coord3D inter_norm = (proj_pos - spheres[tId].center) / (proj_pos - spheres[tId].center).norm();
									Coord3D p1 = proj_pos;
									Coord3D p2 = spheres[tId].center + inter_norm * spheres[tId].radius;
									Real inter_dist = spheres[tId].radius - (proj_pos - spheres[tId].center).norm();

									if (inter_dist < 0) inter_dist = 0;

									//nbr_out.setElement(tId, cnt, j);
									//int idx_con = nbr_out.getElementIndex(tId, cnt);
									//list_j.insert(j);
									int idx_con = prefix[tId] + cnt;
									nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_FLUID_STICKINESS, p1, p2, inter_norm, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = inter_dist;
									cnt++;
								}
							}
							else
							{
								int idx;
								Real max_dist = 0.0f;
								for (int iii = 0; iii < 4; iii++)
								{
									if (tets_sdf[4 * (j - elementOffset.tetOffset) + iii] > max_dist)
									{
										max_dist = tets_sdf[4 * (j - elementOffset.tetOffset) + iii];
										idx = iii;
									}
								}
								int id1 = (idx + 1) % 4;
								int id2 = (idx + 2) % 4;
								int id3 = (idx + 3) % 4;
								Triangle3D t3d(
									tets[j - elementOffset.tetOffset].v[id1],
									tets[j - elementOffset.tetOffset].v[id2],
									tets[j - elementOffset.tetOffset].v[id3]);
								proj_pos = pos_sphere.project(t3d).origin;

								if (pos_sphere.inside(tets[j - elementOffset.tetOffset]))
								{


									Coord3D inter_norm = -(proj_pos - spheres[tId].center) / (proj_pos - spheres[tId].center).norm();
									Coord3D p1 = proj_pos;
									Coord3D p2 = spheres[tId].center - inter_norm * spheres[tId].radius;
									Real inter_dist = spheres[tId].radius + (proj_pos - spheres[tId].center).norm();

									//nbr_out.setElement(tId, cnt, j);
									//int idx_con = nbr_out.getElementIndex(tId, cnt);
									//list_j.insert(j);
									int idx_con = prefix[tId] + cnt;
									nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_FLUID_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = inter_dist;
									cnt++;
								}
							}
						}
						else
						{
							if (pos_sphere.inside(tets[j - elementOffset.tetOffset]))
							{

								Coord3D inter_norm = -(proj_pos - spheres[tId].center) / (proj_pos - spheres[tId].center).norm();
								Coord3D p1 = proj_pos;
								Coord3D p2 = spheres[tId].center - inter_norm * spheres[tId].radius;
								Real inter_dist = spheres[tId].radius + (proj_pos - spheres[tId].center).norm();

								//nbr_out.setElement(tId, cnt, j);
								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;
								nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
								nbr_cons[idx_con].interpenetration = inter_dist;
								cnt++;
							}
							else if ((proj_pos - spheres[tId].center).norm() < spheres[tId].radius + boundary_expand)
							{
								Coord3D inter_norm = (proj_pos - spheres[tId].center) / (proj_pos - spheres[tId].center).norm();
								Coord3D p1 = proj_pos;
								Coord3D p2 = spheres[tId].center + inter_norm * spheres[tId].radius;
								Real inter_dist = spheres[tId].radius - (proj_pos - spheres[tId].center).norm();
								if (inter_dist < 0) inter_dist = 0;

								//nbr_out.setElement(tId, cnt, j);
								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;
								nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
								nbr_cons[idx_con].interpenetration = inter_dist;
								cnt++;
							}
						}
					}

					break;
				}
				case CT_SEG:
				{
					if (filter.sphere_capsule)
					{
						Point3D pos_sphere(spheres[tId].center);
						Segment3D segment_tmp = caps[j - elementOffset.segOffset].segment;
						//printf("ssssssssssssssssssssaaaaaaaaaaaaaaaaa\n");
						if (pos_sphere.distance(segment_tmp) < spheres[tId].radius + caps[j - elementOffset.segOffset].radius + boundary_expand)
						{

							//printf("ssssssssssssssssssss\n");
							//printf("CNT_OKKKK\n");
							/*if( ((j - elementOffset.segOffset) % 39 == 0 || (j - elementOffset.segOffset) % 39 == 38)
								||
							((pos_sphere.project(segment_tmp).origin - pos_sphere.origin).dot(segment_tmp.direction()) < EPSILON))
							*/ {
								Coord3D proj_pos = pos_sphere.project(segment_tmp).origin;
								Coord3D inter_norm = (proj_pos - spheres[tId].center) / (proj_pos - spheres[tId].center).norm();
								Coord3D p1 = proj_pos - inter_norm * caps[j - elementOffset.segOffset].radius;
								Coord3D p2 = spheres[tId].center + inter_norm * spheres[tId].radius;
								Real inter_dist =
									spheres[tId].radius
									+ caps[j - elementOffset.segOffset].radius
									//+ boundary_expand
									- (proj_pos - spheres[tId].center).norm();

								if (inter_dist < 0) inter_dist = 0;

								/*printf("%.5lf %.5lf %.5lf     %.5lf\n",
									inter_norm[0], inter_norm[1], inter_norm[2],
									inter_dist);*/

								//nbr_out.setElement(tId, cnt, j);
								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;
								if (inter_dist > spheres[tId].radius)
								{
									nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_FLUID_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = inter_dist - spheres[tId].radius;
								}
								else
								{
									nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_FLUID_STICKINESS, p1, p2, inter_norm, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = inter_dist;
								}
								cnt++;
							}
						}
					}
					break;
				}
				case CT_TRI:
				{

					Point3D pos_sphere(spheres[tId].center);
					Triangle3D tri_tmp = tris[j - elementOffset.triOffset];
					Real proj_dist = abs(pos_sphere.distance(tri_tmp));
					if (filter.sphere_tri)
						if (proj_dist < spheres[tId].radius + 1.75 * boundary_expand && proj_dist > EPSILON
							//&& ((pos_sphere.project(tri_tmp).origin - pos_sphere.origin) / (pos_sphere.project(tri_tmp).origin - pos_sphere.origin).norm()).dot(tri_tmp.normal() / tri_tmp.normal().norm()) < EPSILON
							&& (((pos_sphere.project(tri_tmp).origin - pos_sphere.origin) / (pos_sphere.project(tri_tmp).origin - pos_sphere.origin).norm()).cross(tri_tmp.normal() / tri_tmp.normal().norm())).norm() < 0.001
							)
						{

							//printf();

							Coord3D proj_pos = pos_sphere.project(tri_tmp).origin;

							Coord3D inter_norm = (proj_pos - spheres[tId].center) / proj_dist;
							Coord3D p1 = proj_pos;
							Coord3D p2 = spheres[tId].center;// + inter_norm * spheres[tId].radius;
							Real inter_dist = spheres[tId].radius - proj_dist + boundary_expand * 1.25;
							if (inter_dist < 0) inter_dist = 0;

							//printf("%.3lf %.3lf %.3lf\n", inter_norm[0], inter_norm[1], inter_norm[2]);

							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;
							nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_FLUID_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = inter_dist;
							if (inter_dist < boundary_expand * 0.25)
								nbr_cons[idx_con].contactType = ContactType::CT_FLUID_STICKINESS;
							cnt++;
							//cnt++;
						}
					break;

				}
				default:
					break;
				}
			}
			break;
		}
		case CT_BOX:
		{
			//printf("YES\n");
			//int nbSize = nbr.getNeighborSize(tId);
			List<int>& list_i = nbr[tId];
			//List<int>& list_j = nbr_out[tId];
			int nbSize = list_i.size();
			for (int ne = 0; ne < nbSize; ne++)
			{
				//int j = nbr.getElement(tId, ne);
				int j = list_i[ne];
				ElementType eleType_j = checkElementType(j, elementOffset);
				/*if (j < 8)
				{
					printf("===================== !!!!!!!!!!!!! %d %d\n", tId, j);
				}*/
				switch (eleType_j)
				{
				case CT_SPHERE:
				{
					if (filter.sphere_box)
					{
						Point3D pos_sphere(spheres[j].center);
						Coord3D proj_pos = pos_sphere.project(boxes[tId - elementOffset.boxOffset]).origin;
						//printf("sphere!!!! === %d\n", j);
						if (pos_sphere.inside(boxes[tId - elementOffset.boxOffset]))
						{
							Coord3D inter_norm = -(proj_pos - spheres[j].center) / (proj_pos - spheres[j].center).norm();
							Coord3D p1 = proj_pos;
							Coord3D p2 = spheres[j].center - inter_norm * spheres[j].radius;
							Real inter_dist = spheres[j].radius + (proj_pos - spheres[j].center).norm();

							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;
							nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
							//nbr_cons[idx_con] = NeighborConstraints(tId, j, 1, p1, p2, 0.0f, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = inter_dist;
							cnt++;
							//printf("sphere!!!! %d\n", j);
						}
						else if ((proj_pos - spheres[j].center).norm() < spheres[j].radius + boundary_expand)
						{
							Coord3D inter_norm = (proj_pos - spheres[j].center) / (proj_pos - spheres[j].center).norm();
							Coord3D p1 = proj_pos;
							Coord3D p2 = spheres[j].center + inter_norm * spheres[j].radius;
							Real inter_dist = spheres[j].radius - (proj_pos - spheres[j].center).norm();
							if (inter_dist < 0) inter_dist = 0;

							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;
							nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_FLUID_STICKINESS, p1, p2, inter_norm, Coord3D(0, 0, 0));
							//nbr_cons[idx_con] = NeighborConstraints(tId, j, 1, p1, p2, 0.0f, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = 0.0f;//inter_dist;
							cnt++;
							//printf("sphere!!!! %d\n", j);
						}
					}
					break;
				}

				case CT_BOX: //boxes
				{
					TManifold<Real> manifold;

					auto boxA = boxes[tId - elementOffset.boxOffset];
					auto boxB = boxes[j - elementOffset.boxOffset];
					CollisionDetection<Real>::request(manifold, boxA, boxB);

					for (int cn = 0; cn < manifold.contactCount; cn++)
					{
						int idx_con = prefix[tId] + cnt;

						NeighborConstraints cPair;

						cPair.pos1 = manifold.contacts[cn].position;
						cPair.pos2 = manifold.contacts[cn].position;
						cPair.normal1 = -manifold.normal;
						cPair.normal2 = manifold.normal;
						cPair.bodyId1 = tId;
						cPair.bodyId2 = j;
						cPair.contactType = ContactType::CT_NONPENETRATION;
						cPair.interpenetration = -manifold.contacts[cn].penetration;
						nbr_cons[idx_con] = cPair;

						cnt += 1;
					}

// 					if (filter.box_box)
// 					{
// 						Coord3D inter_norm1, p11, p21;
// 						Coord3D inter_norm2, p12, p22;
// 						Real inter_dist1;
// 						Real inter_dist2;
// 
// 
// 						int type = 0;
// 						bool insert_one = boxes[tId - elementOffset.boxOffset].point_intersect(boxes[j - elementOffset.boxOffset], inter_norm1, inter_dist1, p11, p21);
// 						bool insert_two = boxes[j - elementOffset.boxOffset].point_intersect(boxes[tId - elementOffset.boxOffset], inter_norm2, inter_dist2, p12, p22);
// 
// 						if (insert_one && insert_two)
// 						{
// 							if (inter_dist1 < inter_dist2) type = 1;
// 							else type = 2;
// 						}
// 						else if (insert_one) type = 1;
// 						else if (insert_two) type = 2;
// 
// 						if (type == 1)
// 						{
// 							//nbr_out.setElement(tId, cnt, j);
// 
// 							/*set up constraints*/
// 							//int idx_con = nbr_out.getElementIndex(tId, cnt);
// 							//list_j.insert(j);
// 							int idx_con = prefix[tId] + cnt;
// 							nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_NONPENETRATION, p11, p21, inter_norm1, Coord3D(0, 0, 0));
// 							nbr_cons[idx_con].interpenetration = -inter_dist1;
// 
// 							//printf("Interpenetration: %f \n", nbr_cons[idx_con].interpenetration);
// 							cnt++;
// 						}
// 						else if (type == 2)
// 						{
// 
// 							//nbr_out.setElement(tId, cnt, j);
// 
// 							/*set up constraints*/
// 							//int idx_con = nbr_out.getElementIndex(tId, cnt);
// 							//list_j.insert(j);
// 							int idx_con = prefix[tId] + cnt;
// 							nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_NONPENETRATION, p12, p22, inter_norm2, Coord3D(0, 0, 0));
// 							nbr_cons[idx_con].interpenetration = -inter_dist2;
// 
// 							//printf("Interpenetration: %f \n", nbr_cons[idx_con].interpenetration);
// 
// 							cnt++;
// 						}
// 					}
					break;
				}
				case CT_TET:// tets
				{
					//printf("BBBBBBBBBBBBBBBBBBTTTTTTTTTTT\n");
					if (filter.box_tet)
					{
						Coord3D inter_norm, p1, p2;
						Real inter_dist;

						if (boxes[tId - elementOffset.boxOffset].point_intersect(tets[j - elementOffset.tetOffset], inter_norm, inter_dist, p1, p2))
						{
							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;
							nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = -inter_dist;
							cnt++;
						}
					}
					break;
				}
				case CT_SEG:// segments
				{
					//printf("YYYYYYYYYEEEEEEEEESSSSSSSSSSSSS %d", j);
					if (filter.box_capsule)
					{
						Segment3D segment_tmp = caps[j - elementOffset.segOffset].segment;
						Segment3D segment_intersect;
						Segment3D segment_prox = segment_tmp.proximity(boxes[tId - elementOffset.boxOffset]);
						if (segment_tmp.intersect(boxes[tId - elementOffset.boxOffset], segment_intersect))
						{
							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;
							Coord3D p1, p2;
							Real interDist = 0.0f;
							Point3D inp((segment_intersect.startPoint() + segment_intersect.endPoint()) / 2.0f);
							Point3D sp(segment_intersect.startPoint());
							Point3D ep(segment_intersect.endPoint());
							if (abs(inp.distance(boxes[tId - elementOffset.boxOffset])) > abs(interDist))
							{
								p2 = inp.origin;
								p1 = inp.project(boxes[tId - elementOffset.boxOffset]).origin;
								interDist = -abs(inp.distance(boxes[tId - elementOffset.boxOffset]));
							}
							if (abs(sp.distance(boxes[tId - elementOffset.boxOffset])) > abs(interDist))
							{
								p2 = sp.origin;
								p1 = sp.project(boxes[tId - elementOffset.boxOffset]).origin;
								interDist = -abs(sp.distance(boxes[tId - elementOffset.boxOffset]));
							}
							if (abs(ep.distance(boxes[tId - elementOffset.boxOffset])) > abs(interDist))
							{
								p2 = ep.origin;
								p1 = ep.project(boxes[tId - elementOffset.boxOffset]).origin;
								interDist = -abs(ep.distance(boxes[tId - elementOffset.boxOffset]));
							}

							nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_NONPENETRATION, p1, p2, (p2 - p1) / (p1 - p2).norm(), Coord3D(0, 0, 0));


							nbr_cons[idx_con].interpenetration = -interDist + caps[j - elementOffset.segOffset].radius;
							
							cnt++;
						}
						else if (segment_prox.length() < caps[j - elementOffset.segOffset].radius)
						{
							//if (segment_prox.direction().dot(segment_tmp.direction()) < EPSILON)//////to delete
							{
								//nbr_out.setElement(tId, cnt, j);
								Coord3D p1, p2;
								Real interDist = caps[j - elementOffset.segOffset].radius - segment_prox.length();
								p1 = segment_prox.v1;
								p2 = segment_prox.v0;

								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;
								nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_NONPENETRATION, p1, p2, (p1 - p2) / (p1 - p2).norm(), Coord3D(0, 0, 0));
								nbr_cons[idx_con].interpenetration = interDist;
								cnt++;
							}
						}
					}
					break;
				}
				case CT_TRI:// tris
				{
					//printf("BBBBBBBBBBBBBBBBBBTTTTTTTTTTT\n");
					if (filter.box_tri)
					{
						Coord3D inter_norm, p1, p2;
						Real inter_dist;
						//printf("AABox!!!!!!!!!!!!!!!!!!!\n");
						if (boxes[tId - elementOffset.boxOffset].point_intersect(tris[j - elementOffset.triOffset],
							inter_norm, inter_dist, p1, p2))
						{
							//printf("ABox!!!!!!!!!!!!!!!!!!!\n");
							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;
							nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = -inter_dist;
							cnt++;
						}
					}
					break;
				}
				default:
				{
					break;
				}
				}
			}
			break;
		}

		case CT_TET:
		{
			//printf("YYYYYYYYYYYEEEEEEEEEEESSSSSSSSSSS\n");
			//int nbSize = nbr.getNeighborSize(tId);
			List<int>& list_i = nbr[tId];
			//List<int>& list_j = nbr_out[tId];
			int nbSize = list_i.size();

			for (int ne = 0; ne < nbSize; ne++)
			{
				//int j = nbr.getElement(tId, ne);
				int j = list_i[ne];
				ElementType eleType_j = checkElementType(j, elementOffset);
				switch (eleType_j)
				{
				case CT_SPHERE:
				{
					if (filter.sphere_tet)
					{
						if (filter.tet_sdf)// && tets_sdf.size() > 0)
						{
							Point3D pos_sphere(spheres[j].center);
							Bool tmp;
							int idx;
							Coord3D proj_pos = pos_sphere.project(tets[tId - elementOffset.tetOffset], tmp, &idx).origin;

							int id1 = (idx + 1) % 4;
							int id2 = (idx + 2) % 4;
							int id3 = (idx + 3) % 4;

							Triangle3D t3d_tmp = Triangle3D(
								tets[tId - elementOffset.tetOffset].v[id1],
								tets[tId - elementOffset.tetOffset].v[id2],
								tets[tId - elementOffset.tetOffset].v[id3]
							);

							if (tets_sdf[4 * (tId - elementOffset.tetOffset) + id1] < EPSILON &&
								tets_sdf[4 * (tId - elementOffset.tetOffset) + id2] < EPSILON &&
								tets_sdf[4 * (tId - elementOffset.tetOffset) + id3] < EPSILON)
							{
								if (pos_sphere.inside(tets[tId - elementOffset.tetOffset]))
								{
									Coord3D inter_norm = -(proj_pos - spheres[j].center) / (proj_pos - spheres[j].center).norm();
									Coord3D p1 = proj_pos;
									Coord3D p2 = spheres[j].center - inter_norm * spheres[j].radius;
									Real inter_dist = spheres[j].radius + (proj_pos - spheres[j].center).norm();

									//nbr_out.setElement(tId, cnt, j);
									//int idx_con = nbr_out.getElementIndex(tId, cnt);
									//list_j.insert(j);
									int idx_con = prefix[tId] + cnt;
									nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_FLUID_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = inter_dist;
									cnt++;
								}
								else if ((proj_pos - spheres[j].center).norm() < spheres[j].radius + boundary_expand
									&& (((proj_pos - spheres[j].center) / (proj_pos - spheres[j].center).norm()).cross(t3d_tmp.normal() / t3d_tmp.normal().norm())).norm() < EPSILON)
								{
									//printf("YYEESS\n");
									Coord3D inter_norm = (proj_pos - spheres[j].center) / (proj_pos - spheres[j].center).norm();
									Coord3D p1 = proj_pos;
									Coord3D p2 = spheres[j].center + inter_norm * spheres[j].radius;
									Real inter_dist = spheres[j].radius - (proj_pos - spheres[j].center).norm();
									if (inter_dist < 0) inter_dist = 0;

									//nbr_out.setElement(tId, cnt, j);
									//int idx_con = nbr_out.getElementIndex(tId, cnt);
									//list_j.insert(j);
									int idx_con = prefix[tId] + cnt;
									nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_FLUID_STICKINESS, p1, p2, inter_norm, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = inter_dist;
									cnt++;
								}
							}
							else
							{
								int idx;
								Real max_dist = 0.0f;
								for (int iii = 0; iii < 4; iii++)
								{
									if (tets_sdf[4 * (tId - elementOffset.tetOffset) + iii] > max_dist)
									{
										max_dist = tets_sdf[4 * (tId - elementOffset.tetOffset) + iii];
										idx = iii;
									}
								}
								int id1 = (idx + 1) % 4;
								int id2 = (idx + 2) % 4;
								int id3 = (idx + 3) % 4;
								Triangle3D t3d(
									tets[j - elementOffset.tetOffset].v[id1],
									tets[j - elementOffset.tetOffset].v[id2],
									tets[j - elementOffset.tetOffset].v[id3]);
								proj_pos = pos_sphere.project(t3d).origin;
								if (pos_sphere.inside(tets[tId - elementOffset.tetOffset]))
								{
									Coord3D inter_norm = -(proj_pos - spheres[j].center) / (proj_pos - spheres[j].center).norm();
									Coord3D p1 = proj_pos;
									Coord3D p2 = spheres[j].center - inter_norm * spheres[j].radius;
									Real inter_dist = spheres[j].radius + (proj_pos - spheres[j].center).norm();

									//nbr_out.setElement(tId, cnt, j);
									//int idx_con = nbr_out.getElementIndex(tId, cnt);
									//list_j.insert(j);
									int idx_con = prefix[tId] + cnt;
									nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_FLUID_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = inter_dist;
									cnt++;
								}
							}
						}
						else
						{
							Point3D pos_sphere(spheres[j].center);
							Bool tmp;
							int idx;
							Coord3D proj_pos = pos_sphere.project(tets[tId - elementOffset.tetOffset], tmp, &idx).origin;
							if (pos_sphere.inside(tets[tId - elementOffset.tetOffset]))
							{
								Coord3D inter_norm = -(proj_pos - spheres[j].center) / (proj_pos - spheres[j].center).norm();
								Coord3D p1 = proj_pos;
								Coord3D p2 = spheres[j].center - inter_norm * spheres[j].radius;
								Real inter_dist = spheres[j].radius + (proj_pos - spheres[j].center).norm();

								//nbr_out.setElement(tId, cnt, j);
								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;
								nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
								nbr_cons[idx_con].interpenetration = inter_dist;
								cnt++;
							}
							else if ((proj_pos - spheres[j].center).norm() < spheres[j].radius + boundary_expand)
							{
								Coord3D inter_norm = (proj_pos - spheres[j].center) / (proj_pos - spheres[j].center).norm();
								Coord3D p1 = proj_pos;
								Coord3D p2 = spheres[j].center + inter_norm * spheres[j].radius;
								Real inter_dist = spheres[j].radius - (proj_pos - spheres[j].center).norm();
								if (inter_dist < 0) inter_dist = 0;

								//nbr_out.setElement(tId, cnt, j);
								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;
								nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
								nbr_cons[idx_con].interpenetration = inter_dist;
								cnt++;
							}
						}
					}
					break;
				}
				case CT_BOX:
				{

					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if (filter.box_tet)
						if (boxes[j - elementOffset.boxOffset].point_intersect(tets[tId - elementOffset.tetOffset], inter_norm, inter_dist, p1, p2))
						{

							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;
							nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = -inter_dist;
							cnt++;
							//printf("TTTTTTTTTTTBBBBBBBBBBBBBBBBBB interDist = %.3lf   interNormal = %.3lf %.3lf %.3lf\n",
							//inter_dist,
							//inter_norm[0], inter_norm[1], inter_norm[2]);
						}

					break;
				}
				case CT_TET:
				{
					if (filter.tet_tet)
					{

						if (filter.tet_neighbor_filter
							&& tet_body_ids.size() > 0
							&& tet_body_ids[tId - elementOffset.tetOffset] == tet_body_ids[j - elementOffset.tetOffset])
						{
							bool tmp = false;
							for (int iii = 0; iii < 4; iii++)
							{ 
								for (int jjj = 0; jjj < 4; jjj++)
									if (tet_element_ids[tId - elementOffset.tetOffset][iii]
										== tet_element_ids[j - elementOffset.tetOffset][jjj])
									{
										tmp = true;
										break;
									}
								if (tmp)
									break;
							}
							if (tmp)
								break;

						}

						Coord3D inter_norm1, p11, p21;
						Coord3D inter_norm2, p12, p22;
						Real inter_dist1;
						Real inter_dist2;



						int type = 0;
						bool insert_one = tets[tId - elementOffset.tetOffset].intersect(tets[j - elementOffset.tetOffset], inter_norm1, inter_dist1, p11, p21);
						bool insert_two = tets[j - elementOffset.tetOffset].intersect(tets[tId - elementOffset.tetOffset], inter_norm2, inter_dist2, p12, p22);


						Coord3D tet_center1 = (tets[tId - elementOffset.tetOffset].v[0]
							+ tets[tId - elementOffset.tetOffset].v[1]
							+ tets[tId - elementOffset.tetOffset].v[2]
							+ tets[tId - elementOffset.tetOffset].v[3])
							/ 4.0f;
						Coord3D tet_center2 = (tets[j - elementOffset.tetOffset].v[0]
							+ tets[j - elementOffset.tetOffset].v[1]
							+ tets[j - elementOffset.tetOffset].v[2]
							+ tets[j - elementOffset.tetOffset].v[3])
							/ 4.0f;
						if (insert_one && insert_two)
						{
							if (inter_dist1 < inter_dist2 && inter_dist1 < - 3 * EPSILON) type = 1;
							else if(inter_dist2 < - 3 * EPSILON) type = 2;
						}
						else if (insert_one && inter_dist1 < -3 * EPSILON) type = 1;
						else if (insert_two && inter_dist2 < -3 * EPSILON) type = 2;

						ContactType ctype = ContactType::CT_FLUID_NONPENETRATION;

						if (type == 1)
						{
							//list_j.insert(j);
							//int idx_con = prefix[tId] + cnt;
							/*set up constraints*/
							if (abs(inter_dist1) < EPSILON)
							{
								inter_norm1 = (tet_center1 - tet_center2) / (tet_center1 - tet_center2).norm();
								ctype = ContactType::CT_LOACL_NONPENETRATION;
							}

							int idx;
							Real max_dist = 0.0f;
							for (int iii = 0; iii < 4; iii++)
							{
								
								if (tets_sdf.size() > 0)
									if (tets_sdf[4 * (j - elementOffset.tetOffset) + iii] > max_dist)
									{

										max_dist = tets_sdf[4 * (j - elementOffset.tetOffset) + iii];
										idx = iii;
									}
							}
							int id1 = (idx + 1) % 4;
							int id2 = (idx + 2) % 4;
							int id3 = (idx + 3) % 4;
							Triangle3D t3d(
								tets[j - elementOffset.tetOffset].v[id1],
								tets[j - elementOffset.tetOffset].v[id2],
								tets[j - elementOffset.tetOffset].v[id3]);
							Coord3D proj_pos = Point3D(p11).project(t3d).origin;


							if (max_dist < EPSILON || abs(inter_dist1) < EPSILON)
							{ 
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;
								if (idx_con < nbr_cons.size())
								{
									nbr_cons[idx_con] = NeighborConstraints(tId, j, ctype, p11, p21, inter_norm1, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = -inter_dist1;
								}
							}
							else
							{
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;
								if (idx_con < nbr_cons.size())
								{
									nbr_cons[idx_con] = NeighborConstraints(tId, j, ctype, p11, proj_pos,
										(proj_pos - p11) / (proj_pos - p11).norm(), Coord3D(0, 0, 0));

									nbr_cons[idx_con].interpenetration = (proj_pos - p11).norm();
								}
							}

							cnt++;

						}
						else if (type == 2)
						{
							//list_j.insert(j);
							//int idx_con = prefix[tId] + cnt;
							if (abs(inter_dist2) < EPSILON)
							{
								inter_norm2 = (tet_center2 - tet_center1) / (tet_center1 - tet_center2).norm();
								ctype = ContactType::CT_LOACL_NONPENETRATION;
							}
							/*set up constraints*/
							int idx;
							Real max_dist = 0.0f;
							for (int iii = 0; iii < 4; iii++)
							{
								if(tets_sdf.size() > 0)
									if (tets_sdf[4 * (tId - elementOffset.tetOffset) + iii] > max_dist)
									{
										max_dist = tets_sdf[4 * (tId - elementOffset.tetOffset) + iii];
										idx = iii;
									}
							}
							int id1 = (idx + 1) % 4;
							int id2 = (idx + 2) % 4;
							int id3 = (idx + 3) % 4;
							Triangle3D t3d(
								tets[tId - elementOffset.tetOffset].v[id1],
								tets[tId - elementOffset.tetOffset].v[id2],
								tets[tId - elementOffset.tetOffset].v[id3]);
							Coord3D proj_pos = Point3D(p12).project(t3d).origin;


							if (max_dist < EPSILON || abs(inter_dist1) < EPSILON)
							{
								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;
								if(idx_con < nbr_cons.size())
								{ 
									nbr_cons[idx_con] = NeighborConstraints(j, tId, ctype, p12, p22, inter_norm2, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = -inter_dist2;
								}
								//printf("%.5lf\n", inter_dist2);
							}
							else
							{
								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;
								if(idx_con < nbr_cons.size())
								{ 
									nbr_cons[idx_con] = NeighborConstraints(j, tId, ctype, p12, proj_pos,
										(proj_pos - p12) / (proj_pos - p12).norm(), Coord3D(0, 0, 0));

									nbr_cons[idx_con].interpenetration = (proj_pos - p12).norm();
								}
							}
							cnt++;
						}
						
					}
					break;
				}
				case CT_SEG:
				{
					if (filter.tet_capsule)
					{
						Segment3D segment_tmp = caps[j - elementOffset.segOffset].segment;
						Line3D line_tmp(segment_tmp.v0, segment_tmp.direction());
						Segment3D segment_tmp2 = segment_tmp.proximity(tets[tId - elementOffset.tetOffset]);
						Segment3D segment_intersect;
						bool intersect1 = false;
						if (line_tmp.intersect(tets[tId - elementOffset.tetOffset], segment_intersect))
						{
							Real left = (segment_intersect.v0 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							Real right = (segment_intersect.v1 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							if (right < left)
							{
								Real tmp = left;
								left = right;
								right = tmp;
							}
							Real maxx = (segment_tmp.v1 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							if (right < 0 || left > maxx)
							{

							}
							else
							{
								intersect1 = true;
								//nbr_out.setElement(tId, cnt, j);
								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;

								left = max(left, 0.0f);
								right = min(right, maxx);


								Coord3D p1, p2;
								Real interDist = 0.0f;
								Bool tmp_bool;

								Coord3D p11 = segment_tmp.v0 + ((left + right) / 2.0f * segment_tmp.direction().normalize());
								Coord3D p22 = Point3D(p11).project(tets[tId - elementOffset.tetOffset], tmp_bool).origin;//


								if ((p11 - p22).norm() > abs(interDist))
								{
									interDist = -(p11 - p22).norm();
									p1 = p11;
									p2 = p22;
								}
								p11 = segment_tmp.v0 + left * segment_tmp.direction().normalize();
								p22 = Point3D(p11).project(tets[tId - elementOffset.tetOffset], tmp_bool).origin;
								if ((p11 - p22).norm() > abs(interDist))
								{
									p1 = p11; p2 = p22;
									interDist = -(p1 - p2).norm();
								}

								p11 = segment_tmp.v0 + right * segment_tmp.direction().normalize();
								p22 = Point3D(p11).project(tets[tId - elementOffset.tetOffset], tmp_bool).origin;
								if ((p11 - p22).norm() > abs(interDist))
								{
									p1 = p11; p2 = p22;
									interDist = -(p1 - p2).norm();
								}

								nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_NONPENETRATION, p1, p2, (p2 - p1) / (p1 - p2).norm(), Coord3D(0, 0, 0));
								nbr_cons[idx_con].interpenetration = -interDist + caps[j - elementOffset.segOffset].radius;
								cnt++;
							}
						}
						if (!intersect1)
						{
							if (segment_tmp2.length() < caps[j - elementOffset.segOffset].radius)
							{
								//nbr_out.setElement(tId, cnt, j);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;

								Coord3D p1, p2;
								Real interDist = caps[j - elementOffset.segOffset].radius - segment_tmp2.length();
								p1 = segment_tmp2.v1;
								p2 = segment_tmp2.v0;

								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_NONPENETRATION, p1, p2, (p1 - p2) / (p1 - p2).norm(), Coord3D(0, 0, 0));
								nbr_cons[idx_con].interpenetration = interDist;
								cnt++;
							}
						}
					}
					break;
				}
				case CT_TRI:
				{

					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if (filter.tet_tri)
						if (tets[tId - elementOffset.boxOffset].intersect(tris[j - elementOffset.triOffset], inter_norm,
							inter_dist, p1, p2))
						{

							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;

							nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = -inter_dist;
							cnt++;

						}

					break;
				}
				default:
				{
					break;
				}
				}

			}
			break;
		}
		case CT_SEG:
		{
			//int nbSize = nbr.getNeighborSize(tId);
			List<int>& list_i = nbr[tId];
			//List<int>& list_j = nbr_out[tId];
			int nbSize = list_i.size();
			for (int ne = 0; ne < nbSize; ne++)
			{
				//int j = nbr.getElement(tId, ne);
				int j = list_i[ne];

				ElementType eleType_j = checkElementType(j, elementOffset);

				switch (eleType_j)
				{
				case CT_SPHERE:
				{
					if (filter.sphere_capsule)
					{
						Point3D pos_sphere(spheres[j].center);
						Segment3D segment_tmp = caps[tId - elementOffset.segOffset].segment;

						if (pos_sphere.distance(segment_tmp) < spheres[j].radius + caps[tId - elementOffset.segOffset].radius + boundary_expand)
						{

							//printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
							/*if (((tId - elementOffset.segOffset) % 39 == 0 || (tId - elementOffset.segOffset) % 39 == 38)
								||
								((pos_sphere.project(segment_tmp).origin - pos_sphere.origin).dot(segment_tmp.direction()) < EPSILON))
							*/ {
								Coord3D proj_pos = pos_sphere.project(segment_tmp).origin;
								Coord3D inter_norm = (proj_pos - spheres[j].center) / (proj_pos - spheres[j].center).norm();
								Coord3D p1 = proj_pos - inter_norm * caps[tId - elementOffset.segOffset].radius;
								Coord3D p2 = spheres[j].center + inter_norm * spheres[j].radius;
								Real inter_dist =
									spheres[j].radius
									+ caps[tId - elementOffset.segOffset].radius
									- (proj_pos - spheres[j].center).norm();

								if (inter_dist < 0) inter_dist = 0;
								/*printf("%.5lf %.5lf %.5lf     %.5lf\n",
									inter_norm[0], inter_norm[1], inter_norm[2],
									inter_dist);*/

								//nbr_out.setElement(tId, cnt, j);
								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;

								if (inter_dist > spheres[j].radius)
								{
									nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_FLUID_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = inter_dist - spheres[j].radius;//inter_dist;
								}
								else
								{
									nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_FLUID_STICKINESS, p1, p2, inter_norm, Coord3D(0, 0, 0));
									nbr_cons[idx_con].interpenetration = inter_dist;//inter_dist;
								}
								cnt++;
							}
						}
					}
					break;
				}
				case CT_BOX:// segments
				{
					//printf("YYYYYYYYYEEEEEEEEESSSSSSSSSSSSS2222 %d", j);
					if (filter.box_capsule)
					{
						Segment3D segment_tmp = caps[tId - elementOffset.segOffset].segment;
						Segment3D segment_intersect;
						Segment3D segment_prox = segment_tmp.proximity(boxes[j - elementOffset.boxOffset]);
						if (segment_tmp.intersect(boxes[j - elementOffset.boxOffset], segment_intersect))
						{
							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;

							Coord3D p1, p2;
							Real interDist = 0.0f;
							Point3D inp((segment_intersect.startPoint() + segment_intersect.endPoint()) / 2.0f);
							Point3D sp(segment_intersect.startPoint());
							Point3D ep(segment_intersect.endPoint());
							if (abs(inp.distance(boxes[j - elementOffset.boxOffset])) > abs(interDist))
							{
								p2 = inp.origin;
								p1 = inp.project(boxes[j - elementOffset.boxOffset]).origin;
								interDist = -abs(inp.distance(boxes[j - elementOffset.boxOffset]));
							}
							if (abs(sp.distance(boxes[j - elementOffset.boxOffset])) > abs(interDist))
							{
								p2 = sp.origin;
								p1 = sp.project(boxes[j - elementOffset.boxOffset]).origin;
								interDist = -abs(sp.distance(boxes[j - elementOffset.boxOffset]));
							}
							if (abs(ep.distance(boxes[j - elementOffset.boxOffset])) > abs(interDist))
							{
								p2 = ep.origin;
								p1 = ep.project(boxes[j - elementOffset.boxOffset]).origin;
								interDist = -abs(ep.distance(boxes[j - elementOffset.boxOffset]));
							}

							nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_NONPENETRATION, p1, p2, (p2 - p1) / (p1 - p2).norm(), Coord3D(0, 0, 0));


							nbr_cons[idx_con].interpenetration = -interDist + caps[tId - elementOffset.segOffset].radius;
							/*printf(" ================ %d %d %.13lf %.13lf %.13lf   %.13lf %.13lf %.13lf %.13lf\n",
								tId, j,
								segment_tmp.v0[0], segment_tmp.v0[1], segment_tmp.v0[2],
								segment_tmp.v1[0], segment_tmp.v1[1], segment_tmp.v1[2],
								segment_intersect.length());*/
							cnt++;
						}
						else if (segment_prox.length() < caps[tId - elementOffset.segOffset].radius)
						{
							//if (segment_prox.direction().dot(segment_tmp.direction()) < EPSILON)//////to delete
							{
								//nbr_out.setElement(tId, cnt, j);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;

								Coord3D p1, p2;
								Real interDist = caps[tId - elementOffset.segOffset].radius - segment_prox.length();
								p1 = segment_prox.v1;
								p2 = segment_prox.v0;

								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_NONPENETRATION, p1, p2, (p1 - p2) / (p1 - p2).norm(), Coord3D(0, 0, 0));
								nbr_cons[idx_con].interpenetration = interDist;
								cnt++;
							}
						}
					}
					break;
				}
				case CT_TET:
				{
					if (filter.tet_capsule)
					{
						Segment3D segment_tmp = caps[tId - elementOffset.segOffset].segment;
						Line3D line_tmp(segment_tmp.v0, segment_tmp.direction());
						Segment3D segment_tmp2 = segment_tmp.proximity(tets[j - elementOffset.tetOffset]);
						Segment3D segment_intersect;
						bool intersect1 = false;
						if (line_tmp.intersect(tets[j - elementOffset.tetOffset], segment_intersect))
						{
							Real left = (segment_intersect.v0 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							Real right = (segment_intersect.v1 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							if (right < left)
							{
								Real tmp = left;
								left = right;
								right = tmp;
							}
							Real maxx = (segment_tmp.v1 - segment_tmp.v0).dot(segment_tmp.direction().normalize());
							if (right < 0 || left > maxx)
							{

							}
							else
							{
								intersect1 = true;
								//nbr_out.setElement(tId, cnt, j);
								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;

								left = max(left, 0.0f);
								right = min(right, maxx);


								Coord3D p1, p2;
								Real interDist = 0.0f;
								Bool tmp_bool;

								Coord3D p11 = segment_tmp.v0 + ((left + right) / 2.0f * segment_tmp.direction().normalize());
								Coord3D p22 = Point3D(p11).project(tets[j - elementOffset.tetOffset], tmp_bool).origin;//


								if ((p11 - p22).norm() > abs(interDist))
								{
									interDist = -(p11 - p22).norm();
									p1 = p11;
									p2 = p22;
								}
								p11 = segment_tmp.v0 + left * segment_tmp.direction().normalize();
								p22 = Point3D(p11).project(tets[j - elementOffset.tetOffset], tmp_bool).origin;
								if ((p11 - p22).norm() > abs(interDist))
								{
									p1 = p11; p2 = p22;
									interDist = -(p1 - p2).norm();
								}

								p11 = segment_tmp.v0 + right * segment_tmp.direction().normalize();
								p22 = Point3D(p11).project(tets[j - elementOffset.tetOffset], tmp_bool).origin;
								if ((p11 - p22).norm() > abs(interDist))
								{
									p1 = p11; p2 = p22;
									interDist = -(p1 - p2).norm();
								}

								nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_NONPENETRATION, p1, p2, (p2 - p1) / (p1 - p2).norm(), Coord3D(0, 0, 0));
								nbr_cons[idx_con].interpenetration = -interDist + caps[tId - elementOffset.segOffset].radius;
								cnt++;
							}
						}
						if (!intersect1)
						{
							if (segment_tmp2.length() < caps[tId - elementOffset.segOffset].radius)
							{
								//nbr_out.setElement(tId, cnt, j);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;

								Coord3D p1, p2;
								Real interDist = caps[tId - elementOffset.segOffset].radius - segment_tmp2.length();
								p1 = segment_tmp2.v1;
								p2 = segment_tmp2.v0;

								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_NONPENETRATION, p1, p2, (p1 - p2) / (p1 - p2).norm(), Coord3D(0, 0, 0));
								nbr_cons[idx_con].interpenetration = interDist;
								cnt++;
							}
						}
					}
					break;
				}
				case CT_SEG:
				{
					if (filter.capsule_capsule)
					{
						Segment3D segment_1 = caps[j - elementOffset.segOffset].segment;
						Segment3D segment_2 = caps[tId - elementOffset.segOffset].segment;
						Segment3D proxi = (segment_1.proximity(segment_2));
						if (abs(j - tId) >= 3)
							if (proxi.length() < caps[j - elementOffset.segOffset].radius + caps[tId - elementOffset.segOffset].radius)
							{
								Coord3D p1, p2;
								Real interDist = caps[j - elementOffset.segOffset].radius + caps[tId - elementOffset.segOffset].radius - proxi.length();
								p1 = proxi.v1 - proxi.direction() / proxi.length() * caps[tId - elementOffset.segOffset].radius;
								p2 = proxi.v0 + proxi.direction() / proxi.length() * caps[j - elementOffset.segOffset].radius;
								//????????????????????
								//int idx_con = nbr_out.getElementIndex(tId, cnt);
								//list_j.insert(j);
								int idx_con = prefix[tId] + cnt;

								nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_FLUID_NONPENETRATION, p1, p2, proxi.direction() / proxi.length(), Coord3D(0, 0, 0));
								nbr_cons[idx_con].interpenetration = interDist;
								//cnt++;
								cnt++;
							}
					}
					break;
				}
				case CT_TRI:
				{
					if (filter.capsule_tri)
					{
						Segment3D segment = caps[tId - elementOffset.segOffset].segment;
						Triangle3D triangle = tris[j - elementOffset.triOffset];
						Point3D p3d;
						if (segment.intersect(triangle, p3d))
						{
							Coord3D p1, p2;

							if (p3d.distance(Point3D(segment.v0)) < p3d.distance(Point3D(segment.v1)))
								p1 = segment.v0;
							else
								p1 = segment.v1;

							//p2 = p3d.origin;
							p2 = (Point3D(p1).project(triangle)).origin;

							Real interDist = (p1 - p2).norm();


							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;
							nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_FLUID_NONPENETRATION, p1, p2, (p2 - p1) / interDist, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = 0;//interDist;
							cnt++;
						}
						else if ((segment.proximity(triangle)).length() < caps[tId - elementOffset.segOffset].radius)
						{
							//printf("bbbbbbbb\n");
							Segment3D proxi = segment.proximity(triangle);
							Coord3D p1, p2;
							Real interDist = caps[j - elementOffset.segOffset].radius - (segment.proximity(triangle)).length();
							p1 = proxi.v0;
							p2 = proxi.v1;
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;

							nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_FLUID_NONPENETRATION, p1, p2, -proxi.direction() / proxi.length(), Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = 0;// interDist;
							cnt++;
						}
					}
					break;
				}
				default:
					break;
				}
			}
			break;
		}
		case CT_TRI:
		{

			//printf("TTTRRRRRRRRIIIIIIIII\n");
			//int nbSize = nbr.getNeighborSize(tId);
			List<int>& list_i = nbr[tId];
			//List<int>& list_j = nbr_out[tId];
			int nbSize = list_i.size();

			for (int ne = 0; ne < nbSize; ne++)
			{
				//int j = nbr.getElement(tId, ne);
				int j = list_i[ne];

				ElementType eleType_j = checkElementType(j, elementOffset);

				switch (eleType_j)
				{

				case CT_SPHERE:
				{
					Point3D pos_sphere(spheres[j].center);
					Triangle3D tri_tmp = tris[tId - elementOffset.triOffset];
					Real proj_dist = abs(pos_sphere.distance(tri_tmp));
					if (filter.sphere_tri)
						//if (pos_sphere.distance(tri_tmp) < spheres[j].radius && pos_sphere.distance(tri_tmp) > EPSILON)
						if (proj_dist < spheres[j].radius + 1.75 * boundary_expand && proj_dist > EPSILON
							//&& ((pos_sphere.project(tri_tmp).origin - pos_sphere.origin) / (pos_sphere.project(tri_tmp).origin - pos_sphere.origin).norm()).dot(tri_tmp.normal() / tri_tmp.normal().norm()) < EPSILON
							&& (((pos_sphere.project(tri_tmp).origin - pos_sphere.origin) / (pos_sphere.project(tri_tmp).origin - pos_sphere.origin).norm()).cross(tri_tmp.normal() / tri_tmp.normal().norm())).norm() < 0.001
							)
						{
							Coord3D proj_pos = pos_sphere.project(tri_tmp).origin;

							Coord3D inter_norm = (proj_pos - spheres[j].center) / proj_dist;
							Coord3D p1 = proj_pos;
							Coord3D p2 = spheres[j].center;// + inter_norm * spheres[j].radius;
							Real inter_dist = spheres[j].radius - proj_dist + boundary_expand * 1.25;
							if (inter_dist < 0) inter_dist = 0;

							//printf("%.3lf %.3lf %.3lf\n", inter_norm[0], inter_norm[1], inter_norm[2]);

							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;

							nbr_cons[idx_con] = NeighborConstraints(tId, j, ContactType::CT_FLUID_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = inter_dist;
							if (inter_dist < boundary_expand * 0.25)
								nbr_cons[idx_con].contactType = ContactType::CT_FLUID_STICKINESS;
							cnt++;

						}
					break;
				}
				case CT_BOX:
				{
					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if (filter.box_tri)
					{
						Coord3D inter_norm, p1, p2;
						Real inter_dist;
						//printf("BBBox!!!!!!!!!!!!!!!!!!!\n");
						if (boxes[j - elementOffset.boxOffset].point_intersect(tris[tId - elementOffset.triOffset],
							inter_norm, inter_dist, p1, p2))
						{
							//printf("BBox!!!!!!!!!!!!!!!!!!!\n");
							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);

							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;

							nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = -inter_dist;
							cnt++;
						}
					}
					break;
				}
				case CT_TET:
				{
					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if (filter.tet_tri)
						if (tets[j - elementOffset.boxOffset].intersect(tris[tId - elementOffset.triOffset], inter_norm,
							inter_dist, p1, p2))
						{

							//nbr_out.setElement(tId, cnt, j);
							//int idx_con = nbr_out.getElementIndex(tId, cnt);

							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;

							nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_NONPENETRATION, p1, p2, inter_norm, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = -inter_dist;
							cnt++;

						}
					break;
				}
				case CT_SEG:
				{

					////Segment3D segment_1 = caps[j - elementOffset.segOffset].segment;
					//Segment3D segment = caps[j - elementOffset.segOffset].segment;
					//Triangle3D triangle = triangles[tId - elementOffset.triOffset];
					//Point3D p3d;
					//if (segment.intersect(triangle, p3d))
					//{
					//	cnt++;
					//}
					//else if ((segment.proximity(triangle)).length() < caps[j - elementOffset.segOffset].radius)
					//{
					//	cnt++;
					//}
					if (filter.capsule_tri)
					{
						Segment3D segment = caps[j - elementOffset.segOffset].segment;
						Triangle3D triangle = tris[tId - elementOffset.triOffset];
						Point3D p3d;
						if (segment.intersect(triangle, p3d))
						{
							Coord3D p1, p2;

							if (p3d.distance(Point3D(segment.v0)) < p3d.distance(Point3D(segment.v1)))
								p1 = segment.v0;
							else
								p1 = segment.v1;

							p2 = (Point3D(p1).project(triangle)).origin;//p1.origin;

							Real interDist = (p1 - p2).norm();


							//int idx_con = nbr_out.getElementIndex(tId, cnt);
							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;

							nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_FLUID_NONPENETRATION, p1, p2, (p2 - p1) / interDist, Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = 0; interDist;
							cnt++;
						}
						else if ((segment.proximity(triangle)).length() < caps[j - elementOffset.segOffset].radius)
						{
							//printf("aaaaaa\n");
							Segment3D proxi = segment.proximity(triangle);
							Coord3D p1, p2;
							Real interDist = caps[j - elementOffset.segOffset].radius - (segment.proximity(triangle)).length();
							p1 = proxi.v0;
							p2 = proxi.v1;
							//int idx_con = nbr_out.getElementIndex(tId, cnt);

							//list_j.insert(j);
							int idx_con = prefix[tId] + cnt;

							nbr_cons[idx_con] = NeighborConstraints(j, tId, ContactType::CT_FLUID_NONPENETRATION, p1, p2, -proxi.direction() / proxi.length(), Coord3D(0, 0, 0));
							nbr_cons[idx_con].interpenetration = 0;//interDist;
							cnt++;
						}
					}
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
			break;
		}
		default:
			break;
		}
	}

	template<typename TDataType>
	void NeighborElementQuery<TDataType>::compute()
	{
		auto inTopo = this->inDiscreteElements()->getDataPtr();

		if (inTopo->totalSize() > 0)
		{
			Real boundary_expand = 0.0f;
			//printf("=========== ============= INSIDE SELF COLLISION %d\n", discreteSet->getTets().size());
			int t_num = inTopo->totalSize();
			if (m_queriedAABB.size() != t_num)
			{
				m_queriedAABB.resize(t_num);
			}
			if (m_queryAABB.size() != t_num)
			{
				m_queryAABB.resize(t_num);
			}

			ElementOffset elementOffset = inTopo->calculateElementOffset();

			cuExecute(t_num,
				NEQ_SetupAABB,
				m_queriedAABB,
				inTopo->getBoxes(),
				inTopo->getSpheres(),
				inTopo->getTets(),
				inTopo->getCaps(),
				inTopo->getTris(),
				elementOffset,
				boundary_expand);

			m_queryAABB.assign(m_queriedAABB);
			

			Real radius = this->inRadius()->getData();

			m_broadPhaseCD->varGridSizeLimit()->setValue(2 * radius);
			m_broadPhaseCD->setSelfCollision(true);


			/*if (this->outNeighborhood()->getElementCount() != t_num)
			{
				this->outNeighborhood()->setElementCount(t_num);
			}*/

			m_broadPhaseCD->inSource()->setValue(m_queryAABB);
			m_broadPhaseCD->inTarget()->setValue(m_queriedAABB);
			// 
			m_broadPhaseCD->update();
	
			Real zero = 0;
	
			//return;
			DArray<int> mapping_nbr;
			DArray<int> cnt_element;
			
			cnt_element.resize(inTopo->totalSize());
			cnt_element.reset();

			cuExecute(inTopo->totalSize(),
				NEQ_Narrow_Count,
				m_broadPhaseCD->outContactList()->getData(),
				inTopo->getBoxes(),
				inTopo->getSpheres(),
				inTopo->getTets(),
				inTopo->getTetSDF(),
				inTopo->getTetBodyMapping(),
				inTopo->getTetElementMapping(),
				inTopo->getCaps(),
				inTopo->getTris(),
				//nbrNum,
				cnt_element,
				elementOffset,
				Filter,
				boundary_expand);

			if (this->outContacts()->isEmpty())
				this->outContacts()->allocate();

			int sum = m_reduce.accumulate(cnt_element.begin(), cnt_element.size());

			auto& contacts = this->outContacts()->getData();
			m_scan.exclusive(cnt_element, true);
			contacts.resize(sum);
			if (sum > 0)
			{
				cuExecute(inTopo->totalSize(),
					NEQ_Narrow_Set,
					m_broadPhaseCD->outContactList()->getData(),
					inTopo->getBoxes(),
					inTopo->getSpheres(),
					inTopo->getTets(),
					inTopo->getTetSDF(),
					inTopo->getTetBodyMapping(),
					inTopo->getTetElementMapping(),
					inTopo->getCaps(),
					inTopo->getTris(),
					contacts,
					cnt_element,
					elementOffset,
					Filter,
					boundary_expand
				);
			}

			mapping_nbr.clear();
			cnt_element.clear();

		}
		else
		{
			//printf("NeighborElementQuery: Empty discreteSet! \n");
		}
	}

	DEFINE_CLASS(NeighborElementQuery);
}