#pragma once
#include "RigidBodySystem.h"



namespace dyno
{
	typedef typename TOrientedBox3D<Real> Box3D;
	typedef typename Quat<Real> TQuat;

	IMPLEMENT_CLASS_1(RigidBodySystem, TDataType)
	template<typename TDataType>
	RigidBodySystem<TDataType>::RigidBodySystem(std::string name)
		: Node(name)
	{
		m_shapes = std::make_shared<DiscreteElements<TDataType>>();
		this->setTopologyModule(m_shapes);
	}

	template<typename TDataType>
	RigidBodySystem<TDataType>::~RigidBodySystem()
	{
	}

	template <typename Coord, typename Matrix>
	__global__ void RB_initialize_device(
		DArray<Coord> pos,
		DArray<Matrix> rotation,
		DArray<TQuat> rotation_q,
		DArray<Sphere3D> spheres,
		DArray<Box3D> boxes,
		DArray<Tet3D> tets,
		int start_box,
		int start_sphere,
		int start_tet,
		int start_segment,
		int start_mesh
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (pId >= rotation_q.size())
			return;
		rotation_q[pId] = TQuat(0, 0, 0, 1);
		
		if (pId >= start_mesh) return;
		rotation[pId] = Matrix::identityMatrix();
		
		rotation_q[pId] = TQuat(0, 0, 0, 1);
		
		if (pId >= start_segment) {}
		else if (pId >= start_tet) pos[pId] = (tets[pId - start_tet].v[0] + tets[pId - start_tet].v[1] + tets[pId - start_tet].v[2] + tets[pId - start_tet].v[3]) / 4.0f;
		else if (pId >= start_box) pos[pId] = boxes[pId - start_box].center;
		else pos[pId] = spheres[pId - start_sphere].center;
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::resetStates()
	{
		//todo: initialize inertial tensor
		//todo: copy from topology module
		
		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(this->getTopologyModule());//this ???
		m_shapes = TypeInfo::cast<DiscreteElements<DataType3f>>(this->getTopologyModule());
		m_box3d_init.resize(discreteSet->getBoxes().size());
		m_sphere3d_init.resize(discreteSet->getSpheres().size());
		m_tet3d_init.resize(discreteSet->getTets().size());

		//Function1Pt::copy(m_box3d_init, discreteSet->getBoxes());
		m_box3d_init.assign(discreteSet->getBoxes());
		//Function1Pt::copy(m_sphere3d_init, discreteSet->getSpheres());
		m_sphere3d_init.assign(discreteSet->getSpheres());
		//Function1Pt::copy(m_tet3d_init, discreteSet->getTets());
		m_tet3d_init.assign(discreteSet->getTets());
		//printf("@@@");

		int size_rigids = discreteSet->getBoxes().size() + discreteSet->getSpheres().size() + discreteSet->getTets().size();
		int size_rigids_0 = size_rigids;
		size_else = size_rigids;

		start_sphere = 0;
		start_box = discreteSet->getSpheres().size();
		start_tet = discreteSet->getSpheres().size() + discreteSet->getBoxes().size();
		start_segment = discreteSet->getBoxes().size() + discreteSet->getSpheres().size() + discreteSet->getTets().size();

		
		currentRigidRotation()->setElementCount(size_rigids);
		currentAngularVelocity()->setElementCount(size_rigids);
		currentCenter()->setElementCount(size_rigids);
		currentVelocity()->setElementCount(size_rigids);
		currentMass()->setElementCount(size_rigids);
		cnt_boudary.resize(size_rigids);

		m_rotation_q.resize(size_rigids);
		m_inertia.setElementCount(size_rigids);
		m_inertia_init.resize(size_rigids);

		mass_eq.resize(size_rigids * 6);
		mass_buffer.resize(size_rigids * 6);

		uint pDimsR = cudaGridSize(size_rigids, BLOCK_SIZE);
		
		RB_initialize_device << <pDimsR, BLOCK_SIZE >> > (
			currentCenter()->getData(),
			currentRigidRotation()->getData(),
			m_rotation_q,
			m_sphere3d_init,
			m_box3d_init,
			m_tet3d_init,
			start_box,
			start_sphere,
			start_tet,
			start_segment,
			size_rigids_0
			);
			
		cuSynchronize();

		

	
		center_init.resize(size_rigids);
		
		//Function1Pt::copy(center_init, currentCenter()->getData());
		center_init.assign(currentCenter()->getData());

		printf("%d %d %d %d %d\n", currentMass()->getElementCount(), host_inertia_tensor.size(), host_velocity.size(), host_angular_velocity.size(), host_mass.size());

		//Function1Pt::copy(m_inertia.getData(), host_inertia_tensor);
		m_inertia.getData().assign(host_inertia_tensor);

		/*Function1Pt::copy(m_inertia_init, host_inertia_tensor);
		Function1Pt::copy(currentVelocity()->getData(), host_velocity);
		Function1Pt::copy(currentAngularVelocity()->getData(), host_angular_velocity);
		Function1Pt::copy(currentMass()->getData(), host_mass);*/
		m_inertia_init.assign(host_inertia_tensor);
		currentVelocity()->getData().assign(host_velocity);
		currentAngularVelocity()->getData().assign(host_angular_velocity);
		currentMass()->getData().assign(host_mass);

		
		host_inertia_tensor.clear();
		host_velocity.clear();
		host_angular_velocity.clear();
		host_mass.clear();

		//printf("INITIALIZE NEQ\n");
		/* FOR TEST ONLY */
		if(discreteSet->getSize() != 0)
		{ 
			if(m_nbrQueryElement == NULL)
			{ 
				m_nbrQueryElement = this->template addComputeModule<NeighborElementQuery<TDataType>>("neighborhood_rigid");
				m_nbrQueryElement->setDiscreteSet(discreteSet);
				m_nbrQueryElement->initialize();
			}
		}
		//printf("graphics pipline %d\n", this->graphicsPipeline()->isNull())
		this->graphicsPipeline()->initialize();
		//this->graphicsPipeline()->update();
		
		//return true;
	}
	
	template <typename Coord>
	__global__ void RB_update_sphere(
		DArray<Coord> pos,
		DArray<Sphere3D> sphere,
		int start_sphere
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= sphere.size()) return;
		sphere[pId].center = pos[pId + start_sphere];
	}

	template <typename Coord, typename Matrix>
	__global__ void RB_update_box(
		DArray<Coord> pos,
		DArray<Matrix> rotation,
		DArray<Box3D> box,
		DArray<Box3D> box_init,
		int start_box
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= box.size()) return;
		box[pId].center = pos[pId + start_box];

		box[pId].u = rotation[pId + start_box] * box_init[pId].u;
		box[pId].v = rotation[pId + start_box] * box_init[pId].v;
		box[pId].w = rotation[pId + start_box] * box_init[pId].w;

	}

	template <typename Coord, typename Matrix>
	__global__ void RB_update_tet(
		DArray<Coord> pos,
		DArray<Matrix> rotation,
		DArray<Tet3D> tet,
		DArray<Tet3D> tet_init,
		int start_tet
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tet.size()) return;
		//box[pId].center = pos[pId + start_box];
		Coord3D center_init = (tet_init[pId - start_tet].v[0] + tet_init[pId - start_tet].v[1] + tet_init[pId - start_tet].v[2] + tet_init[pId - start_tet].v[3]) / 4.0f;
		tet[pId].v[0] = rotation[pId + start_tet] * (tet_init[pId].v[0] - center_init) + pos[pId + start_tet];
		tet[pId].v[1] = rotation[pId + start_tet] * (tet_init[pId].v[1] - center_init) + pos[pId + start_tet];
		tet[pId].v[2] = rotation[pId + start_tet] * (tet_init[pId].v[2] - center_init) + pos[pId + start_tet];
		tet[pId].v[3] = rotation[pId + start_tet] * (tet_init[pId].v[3] - center_init) + pos[pId + start_tet];
	}

	template <typename Coord, typename Matrix>
	__global__ void RB_update_state(
		DArray<Coord> pos,
		DArray<Matrix> rotation,
		DArray<TQuat> rotation_q,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Matrix> inertia,
		DArray<Matrix> inertia_init,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;
		
		pos[pId] += velocity[pId] * dt;
		
		rotation_q[pId] += dt * 0.5f * 
			TQuat(angular_velocity[pId][0], angular_velocity[pId][1],angular_velocity[pId][2], 0.0f)
			*
			(rotation_q[pId]);
		
		rotation_q[pId] = rotation_q[pId].normalize();
		rotation[pId] = rotation_q[pId].toMatrix3x3();

		inertia[pId] = rotation[pId] * inertia_init[pId] * rotation[pId].inverse();

	}

	template <typename Coord>
	__global__ void RB_update_velocity(
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> AA,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= AA.size() / 2) return;

		
		//printf("%.3lf %.3lf %.3lf\n", AA[2 * pId][0], AA[2 * pId][1], AA[2 * pId][2]);
		velocity[pId] += AA[2 * pId] * dt;// + Coord(0, -9.8f, 0) * dt;
		velocity[pId] += Coord(0, -9.8f, 0) * dt;
		 //printf("velocity: %.3lf %.3lf %.3lf\n", velocity[pId][0], velocity[pId][1], velocity[pId][2]);
		angular_velocity[pId] += AA[2 * pId + 1] * dt;
		

	}

	template <typename Coord, typename Matrix>
	__global__ void RB_constrct_jacobi(
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Real> mass,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<NeighborConstraints> nbc
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbc[pId].idx1;
		int idx2 = nbc[pId].idx2;

		//printf("%d %d\n", idx1, idx2);

		if (nbc[pId].constraint_type == constraint_distance) // test dist constraint
		{
			Coord p2 = nbc[pId].pos2;
			Coord p1 = nbc[pId].pos1;
			Coord d = p2 - p1;
			Coord r1 = p1 - pos[idx1];
			Coord r2 = p2 - pos[idx2];

			J[4 * pId] = -d ;
			J[4 * pId + 1] = (-r1.cross(d));
			J[4 * pId + 2] = d ;
			J[4 * pId + 3] = (r2.cross(d));

			B[4 * pId] = -d / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (- r1.cross(d)) ;
			B[4 * pId + 2] = d / mass[idx2];
			B[4 * pId + 3] = inertia[idx2].inverse() * (r2.cross(d));

			//printf("B: %.3lf %.3lf %.3lf %.3lf\n", B[4 * pId], B[4 * pId + 1], B[4 * pId + 2], B[4 * pId + 3]);

		}
		else if (nbc[pId].constraint_type == constraint_collision) // contact, collision
		{
			Coord p1 = nbc[pId].pos1;
			Coord p2 = nbc[pId].pos2;
			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			Coord r2 = p2 - pos[idx2];

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			J[4 * pId + 2] = -n;
			J[4 * pId + 3] = -(r2.cross(n));

			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			B[4 * pId + 2] = -n / mass[idx2];
			B[4 * pId + 3] = inertia[idx2].inverse() * (-r2.cross(n));
		}

		else if (nbc[pId].constraint_type == constraint_boundary) // boundary
		{
			
			Coord p1 = nbc[pId].pos1;
		//	printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ %d %.3lf %.3lf %.3lf\n", idx1, p1[0], p1[1], p1[2]);

			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			J[4 * pId + 2] = Coord(0);
			J[4 * pId + 3] = Coord(0);

			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			B[4 * pId + 2] = Coord(0);
			B[4 * pId + 3] = Coord(0);
		}

		else if (nbc[pId].constraint_type == constraint_friction) // friction
		{
			Coord p1 = nbc[pId].pos1;
			//printf("~~~~~~~ %.3lf %.3lf %.3lf\n", p1[0], p1[1], p1[2]);

			
			Coord p2 = Coord(0);
			if(idx2 != -1)
				p2 = nbc[pId].pos2;

			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			Coord r2 = Coord(0);
			if (idx2 != -1)
				r2 = p2 - pos[idx2];

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			if(idx2 != -1)
			{ 
				J[4 * pId + 2] = -n;
				J[4 * pId + 3] = -(r2.cross(n));
			}
			else
			{
				J[4 * pId + 2] = Coord(0);
				J[4 * pId + 3] = Coord(0);
			}
			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			if(idx2 != -1)
			{ 
				B[4 * pId + 2] = -n / mass[idx2];
				B[4 * pId + 3] = inertia[idx2].inverse() * (-r2.cross(n));
			}
			else
			{
				B[4 * pId + 2] = Coord(0);
				B[4 * pId + 3] = Coord(0);
			}
		}

	}

	template <typename Coord, typename Matrix>
	__global__ void RB_constrct_jacobi(
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Matrix> inertia_eq,
		DArray<Real> mass,
		DArray<Real> mass_eq,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<NeighborConstraints> nbc
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbc[pId].idx1;
		int idx2 = nbc[pId].idx2;

		//printf("%d %d\n", idx1, idx2);
		//EPSILON

		if (nbc[pId].constraint_type == constraint_collision) // contact, collision
		{
			Coord p1 = nbc[pId].pos1;
			Coord p2 = nbc[pId].pos2;
			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			Coord r2 = p2 - pos[idx2];

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			J[4 * pId + 2] = -n;
			J[4 * pId + 3] = -(r2.cross(n));

			n /= n.norm();
			Coord3D n1(0), n2(0);

			Real ratio1 = 1.0f;
			Real ratio2 = 1.0f;

			if (n[0] > EPSILON)
			{
				n1 += Coord(n[0], 0, 0) / mass_eq[6 * idx1 + 1];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 1];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 1]);
				n2 -= Coord(n[0], 0, 0) / mass_eq[6 * idx2 + 0];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 0];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 0]);

			}
			else
			{
				n1 += Coord(n[0], 0, 0) / mass_eq[6 * idx1 + 0];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 0];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 0]);
				n2 -= Coord(n[0], 0, 0) / mass_eq[6 * idx2 + 1];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 1];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 1]);
			}
			if (n[1] > EPSILON)
			{
				n1 += Coord(0, n[1], 0) / mass_eq[6 * idx1 + 3];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 3];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 3]);
				n2 -= Coord(0, n[1], 0) / mass_eq[6 * idx2 + 2];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 2];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 2]);
			}
			else
			{
				n1 += Coord(0, n[1], 0) / mass_eq[6 * idx1 + 2];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 2];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 2]);
				n2 -= Coord(0, n[1], 0) / mass_eq[6 * idx2 + 3];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 3];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 3]);
			}
			if (n[2] > EPSILON)
			{
				n1 += Coord(0, 0, n[2]) / mass_eq[6 * idx1 + 5];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 5];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 5]);
				n2 -= Coord(0, 0, n[2]) / mass_eq[6 * idx2 + 4];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 4];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 4]);
			}
			else
			{
				n1 += Coord(0, 0, n[2]) / mass_eq[6 * idx1 + 4];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 4];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 4]);
				n2 -= Coord(0, 0, n[2]) / mass_eq[6 * idx2 + 5];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 5];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 5]);
			}

			ratio1 /= 3.0f;
			ratio2 /= 3.0f;

			printf("%.5lf %.5lf %.5lf %.5lf\n", ratio1, ratio2, mass[idx1], mass[idx2]);

			B[4 * pId] = n1;//n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n)) * ratio1;
			B[4 * pId + 2] = n2;//-n / mass[idx2];
			B[4 * pId + 3] = inertia[idx2].inverse() * (-r2.cross(n)) * ratio2;
		}

		else if (nbc[pId].constraint_type == constraint_boundary) // boundary
		{

			Coord p1 = nbc[pId].pos1;
			//	printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ %d %.3lf %.3lf %.3lf\n", idx1, p1[0], p1[1], p1[2]);

			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];


			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			J[4 * pId + 2] = Coord(0);
			J[4 * pId + 3] = Coord(0);

			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			B[4 * pId + 2] = Coord(0);
			B[4 * pId + 3] = Coord(0);
		}

		else if (nbc[pId].constraint_type == constraint_friction) // friction
		{
			Coord p1 = nbc[pId].pos1;
			//printf("~~~~~~~ %.3lf %.3lf %.3lf\n", p1[0], p1[1], p1[2]);


			Coord p2 = Coord(0);
			if (idx2 != -1)
				p2 = nbc[pId].pos2;

			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			Coord r2 = Coord(0);
			if (idx2 != -1)
				r2 = p2 - pos[idx2];

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			if (idx2 != -1)
			{
				J[4 * pId + 2] = -n;
				J[4 * pId + 3] = -(r2.cross(n));
			}
			else
			{
				J[4 * pId + 2] = Coord(0);
				J[4 * pId + 3] = Coord(0);
			}
			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			if (idx2 != -1)
			{
				B[4 * pId + 2] = -n / mass[idx2];
				B[4 * pId + 3] = inertia[idx2].inverse() * (-r2.cross(n));
			}
			else
			{
				B[4 * pId + 2] = Coord(0);
				B[4 * pId + 3] = Coord(0);
			}
		}

	}

	template <typename Coord, typename Matrix, typename Real>
	__global__ void RB_constrct_mass_eq(
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Real> mass,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> mass_eq,
		DArray<Real> mass_eq_old,
		DArray<Matrix> inertia_eq,
		DArray<NeighborConstraints> nbc
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;
		int idx1 = nbc[pId].idx1;
		int idx2 = nbc[pId].idx2;
		if(nbc[pId].constraint_type != constraint_friction)
		{
			Coord d = nbc[pId].normal1;

			if (d[0] > EPSILON)
			{
				Coord3D d_n = d / d.norm();
				if(idx2 != -1)
				{ 
					atomicAdd(&mass_eq[idx1 * 6], mass_eq_old[idx2 * 6] * d_n[0]);
					atomicAdd(&mass_eq[idx2 * 6 + 1], mass_eq_old[idx1 * 6 + 1] * d_n[0]);
				}
				else
				{
					atomicAdd(&mass_eq[idx1 * 6], 100000.0f);
				}
			}
			else
			{
				Coord3D d_n = d / d.norm();
				if (idx2 != -1)
				{
					atomicAdd(&mass_eq[idx1 * 6 + 1], - mass_eq_old[idx2 * 6 + 1] * d_n[0]);
					atomicAdd(&mass_eq[idx2 * 6], - mass_eq_old[idx1 * 6] * d_n[0]);
				}
				else
				{
					atomicAdd(&mass_eq[idx1 * 6], 100000.0f);
				}
			}
			if (d[1] > EPSILON)
			{
				Coord3D d_n = d / d.norm();
				if (idx2 != -1)
				{
					atomicAdd(&mass_eq[idx1 * 6 + 2], mass_eq_old[idx2 * 6 + 2] * d_n[1]);
					atomicAdd(&mass_eq[idx2 * 6 + 3], mass_eq_old[idx1 * 6 + 3] * d_n[1]);
				}
				else
				{
					atomicAdd(&mass_eq[idx1 * 6 + 2], 100000.0f);
				}
			}
			else
			{
				Coord3D d_n = d / d.norm();
				if (idx2 != -1)
				{
					atomicAdd(&mass_eq[idx1 * 6 + 3], - mass_eq_old[idx2 * 6 + 3] * d_n[1]);
					atomicAdd(&mass_eq[idx2 * 6 + 2], - mass_eq_old[idx1 * 6 + 2] * d_n[1]);
				}
				else
				{
					atomicAdd(&mass_eq[idx1 * 6 + 3], 100000.0f);
				}
			}
			if (d[2] > EPSILON)
			{
				Coord3D d_n = d / d.norm();
				if (idx2 != -1)
				{
					atomicAdd(&mass_eq[idx1 * 6 + 4], mass_eq_old[idx2 * 6 + 4] * d_n[2]);
					atomicAdd(&mass_eq[idx2 * 6 + 5], mass_eq_old[idx1 * 6 + 5] * d_n[2]);
				}
				else
				{
					atomicAdd(&mass_eq[idx1 * 6 + 4], 100000.0f);
				}
			}
			else
			{
				Coord3D d_n = d / d.norm();
				if (idx2 != -1)
				{
					atomicAdd(&mass_eq[idx1 * 6 + 5], - mass_eq_old[idx2 * 6 + 5] * d_n[2]);
					atomicAdd(&mass_eq[idx2 * 6 + 4], - mass_eq_old[idx1 * 6 + 4] * d_n[2]);
				}
				else
				{
					atomicAdd(&mass_eq[idx1 * 6 + 5], 100000.0f);
				}
			}

		}
	}

	template <typename Real, typename Matrix>
	__global__ void RB_constrct_mass_eq(
		DArray<Matrix> inertia,
		DArray<Real> mass,
		DArray<Real> mass_eq
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mass.size()) return;

		for (int i = 0; i < 6; i++)
			mass_eq[pId * 6 + i] += mass[pId];
	}


	// ignore zeta !!!!!!
	template <typename Coord>
	__global__ void RB_compute_ita(
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> J,
		DArray<Real> ita,
		DArray<Real> mass,
		DArray<NeighborConstraints> nbq,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbq[pId].idx1;
		int idx2 = nbq[pId].idx2;
		//printf("from ita %d\n", pId);
		Real ita_i = Real(0);
		if (true) // test dist constraint
		{
			ita_i -= J[4 * pId].dot(velocity[idx1]);
			ita_i -= J[4 * pId + 1].dot(angular_velocity[idx1]);
			if(idx2 != -1)
			{ 
				ita_i -= J[4 * pId + 2].dot(velocity[idx2]);
				ita_i -= J[4 * pId + 3].dot(angular_velocity[idx2]);
			}
		}
		ita[pId] = ita_i / dt;
		if (nbq[pId].constraint_type == constraint_collision || nbq[pId].constraint_type == constraint_boundary)
		{
			ita[pId] += min(nbq[pId].inter_distance, nbq[pId].inter_distance) / dt / dt / 15.0f;
		}

	}

	template <typename Coord>
	__global__ void RB_compute_d(
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> D
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;
		Real d = Real(0);
		
		{
			{
				d += J[4 * pId].dot(B[4 * pId]);
				d += J[4 * pId + 1].dot(B[4 * pId + 1]);
				d += J[4 * pId + 2].dot(B[4 * pId + 2]);
				d += J[4 * pId + 3].dot(B[4 * pId + 3]);
			}
		}
		D[pId] = d;
	}

	template <typename Coord>
	__global__ void RB_take_one_iteration(
		DArray<Coord> AA,
		DArray<Real> d,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> ita,
		DArray<Real> lambda,
		DArray<Real> mass,
		DArray<NeighborConstraints> nbq
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbq[pId].idx1;
		int idx2 = nbq[pId].idx2;

		Real ita_i = ita[pId];
		{
			ita_i -= J[4 * pId].dot(AA[idx1 * 2]);
			ita_i -= J[4 * pId + 1].dot(AA[idx1 * 2 + 1]);
			if (idx2 != -1)
			{
				ita_i -= J[4 * pId + 2].dot(AA[idx2 * 2]);
				ita_i -= J[4 * pId + 3].dot(AA[idx2 * 2 + 1]);
			}
		}

		if (d[pId] > EPSILON)
		{
			Real delta_lambda = ita_i / d[pId];
			delta_lambda *= 0.2;

			//printf("delta_lambda = %.3lf\n", delta_lambda);

			if (nbq[pId].constraint_type == constraint_collision || nbq[pId].constraint_type == constraint_boundary) //	PROJECTION!!!!
			{
				Real lambda_new = lambda[pId] + delta_lambda;
				if (lambda_new < 0) lambda_new = 0;

				Real mass_i = mass[idx1];
				if (idx2 != -1)
					mass_i += mass[idx2];

				if (lambda_new > 25 * (mass_i / 0.1)) lambda_new = 25 * (mass_i / 0.1);
				delta_lambda = lambda_new - lambda[pId];
			}

			if (nbq[pId].constraint_type == constraint_friction) //	PROJECTION!!!!
			{
				Real lambda_new = lambda[pId] + delta_lambda;
				

				Real mass_i = mass[idx1];
				if (idx2 != -1)
					mass_i += mass[idx2];



				//if ((lambda_new) > 15 * (mass_i)) lambda_new = 15 * (mass_i);
				//if ((lambda_new) < -15 * (mass_i)) lambda_new = -15 * (mass_i);
				delta_lambda = lambda_new - lambda[pId];
			}

			lambda[pId] += delta_lambda;

			//printf("inside iteration: %d %d %.5lf   %.5lf\n", idx1, idx2, nbq[pId].s4, delta_lambda);

			atomicAdd(&AA[idx1 * 2][0], B[4 * pId][0] * delta_lambda);
			atomicAdd(&AA[idx1 * 2][1], B[4 * pId][1] * delta_lambda);
			atomicAdd(&AA[idx1 * 2][2], B[4 * pId][2] * delta_lambda);

			atomicAdd(&AA[idx1 * 2 + 1][0], B[4 * pId + 1][0] * delta_lambda);
			atomicAdd(&AA[idx1 * 2 + 1][1], B[4 * pId + 1][1] * delta_lambda);
			atomicAdd(&AA[idx1 * 2 + 1][2], B[4 * pId + 1][2] * delta_lambda);

			if (idx2 != -1)
			{
				atomicAdd(&AA[idx2 * 2][0], B[4 * pId + 2][0] * delta_lambda);
				atomicAdd(&AA[idx2 * 2][1], B[4 * pId + 2][1] * delta_lambda);
				atomicAdd(&AA[idx2 * 2][2], B[4 * pId + 2][2] * delta_lambda);

				atomicAdd(&AA[idx2 * 2 + 1][0], B[4 * pId + 3][0] * delta_lambda);
				atomicAdd(&AA[idx2 * 2 + 1][1], B[4 * pId + 3][1] * delta_lambda);
				atomicAdd(&AA[idx2 * 2 + 1][2], B[4 * pId + 3][2] * delta_lambda);

				//AA[idx2 * 2] += B[4 * pId + 2] * delta_lambda;
				//AA[idx2 * 2 + 1] += B[4 * pId + 3] * delta_lambda;
			}
		}

	}
	

	template <typename Coord, typename Matrix> /* FOR TEST */
	__global__ void RB_update_pair_info(
		DArray<Coord> center_init,
		DArray<Coord> center_now,
		DArray<Matrix> rotation,
		DArray<NeighborConstraints> nbq
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbq.size()) return;

		if(nbq[pId].constraint_type == constraint_distance)
		{ 
			int idx1 = nbq[pId].idx1;
			int idx2 = nbq[pId].idx2;
			Coord offset1 = nbq[pId].v1 - center_init[idx1];
			nbq[pId].pos1 = center_now[idx1] + rotation[idx1] * offset1;


			Coord offset2 = nbq[pId].v2 - center_init[idx2];
			nbq[pId].pos2 = center_now[idx2] + rotation[idx2] * offset2;
		}
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::rigid_update_topology()
	{
		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(this->getTopologyModule());


		uint pDimsB = cudaGridSize(m_box3d_init.size(), BLOCK_SIZE);
		if(pDimsB > 0)
		RB_update_box << <pDimsB, BLOCK_SIZE >> > (
			currentCenter()->getData(),
			currentRigidRotation()->getData(),
			discreteSet->getBoxes(),
			m_box3d_init,
			start_box
			);

		cuSynchronize();

		uint pDimsS = cudaGridSize(m_sphere3d_init.size(), BLOCK_SIZE);
		if(pDimsS > 0)
		RB_update_sphere << <pDimsS, BLOCK_SIZE >> > (
			currentCenter()->getData(),
			discreteSet->getSpheres(),
			start_sphere
			);
		cuSynchronize();

		uint pDimsT = cudaGridSize(m_tet3d_init.size(), BLOCK_SIZE);
		if (pDimsT > 0)
			RB_update_tet << <pDimsT, BLOCK_SIZE >> > (
				currentCenter()->getData(),
				currentRigidRotation()->getData(),
				discreteSet->getTets(),
				m_tet3d_init,
				start_tet
				);
		cuSynchronize();

		
		
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::solve_constraint()
	{
		int size_constraints = m_nbrQueryElement->nbr_cons.getElementCount() + buffer_boundary.size();

		if (size_constraints == 0) return;

		for (int it = 0; it < 100; it++)
		{
			// todo : project gs
			uint pDims = cudaGridSize(size_constraints, BLOCK_SIZE);
			RB_take_one_iteration<< <pDims, BLOCK_SIZE >> > (
				AA.getData(),
				D,
				J,
				B,
				ita,
				lambda,
				currentMass()->getData(),
				constraints_all
				);

			cuSynchronize();
		}

	}


	template<typename TDataType>
	void RigidBodySystem<TDataType>::update_position_rotation(Real dt)
	{
		
		uint pDims = cudaGridSize(currentCenter()->getElementCount(), BLOCK_SIZE);
		
		RB_update_velocity << <pDims, BLOCK_SIZE >> > (
			currentVelocity()->getData(),
			currentAngularVelocity()->getData(),
			AA.getData(),
			dt
			);
		cuSynchronize();
		
		RB_update_state << <pDims, BLOCK_SIZE >> > (
			currentCenter()->getData(),
			currentRigidRotation()->getData(),
			m_rotation_q,
			currentVelocity()->getData(),
			currentAngularVelocity()->getData(),
			m_inertia.getData(),
			m_inertia_init,
			dt
			);

		cuSynchronize();


	}

	
	__global__ void RB_update_offset(
		DArray<NeighborConstraints> nbq,
		int offset
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbq.size()) return;
		if(nbq[pId].idx1 != -1)
			nbq[pId].idx1 += offset;
		if(nbq[pId].idx2 != -1)
			nbq[pId].idx2 += offset;
	}



	template <typename Coord>
	__global__ void RB_set_boundary(
		DArray<Sphere3D> sphere,
		DArray<Box3D> box,
		DArray<Tet3D> tet,
		DArray<int> count,
		DArray<NeighborConstraints> nbq,
		Coord hi,
		Coord lo,
		int start_sphere,
		int start_box,
		int start_tet,
		int start_segment
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= sphere.size() + box.size()) return;
		
		if (pId < start_box && pId >= start_sphere)//sphere
		{

		}
		else if (pId >= start_box && pId < start_tet)//box
		{
			//int idx = pId - start_box;
			int cnt = 0;
			int start_i = count[pId];
			Coord center = box[pId].center;
			Coord u = box[pId].u;
			Coord v = box[pId].v;
			Coord w = box[pId].w;
			Coord extent = box[pId].extent;
			Point3D p[8];
			p[0] = Point3D(center - u * extent[0] - v * extent[1] - w * extent[2]);
			p[1] = Point3D(center - u * extent[0] - v * extent[1] + w * extent[2]);
			p[2] = Point3D(center - u * extent[0] + v * extent[1] - w * extent[2]);
			p[3] = Point3D(center - u * extent[0] + v * extent[1] + w * extent[2]);
			p[4] = Point3D(center + u * extent[0] - v * extent[1] - w * extent[2]);
			p[5] = Point3D(center + u * extent[0] - v * extent[1] + w * extent[2]);
			p[6] = Point3D(center + u * extent[0] + v * extent[1] - w * extent[2]);
			p[7] = Point3D(center + u * extent[0] + v * extent[1] + w * extent[2]);
			bool c1, c2, c3, c4, c5, c6;
			c1 = c2 = c3 = c4 = c5 = c6 = true;
			for (int i = 0; i < 8; i++)
			{
				Coord pos = p[i].origin;
				if (pos[0] > hi[0] && c1)
				{
					c1 = true;
					nbq[cnt + start_i].idx1 = pId;
					nbq[cnt + start_i].idx2 = -1;
					nbq[cnt + start_i].normal1 = Coord(-1,0,0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].constraint_type = constraint_boundary;
					nbq[cnt + start_i].inter_distance = pos[0] - hi[0];
					cnt++;
				}
				if (pos[1] > hi[1] && c2)
				{
					c2 = true;
					nbq[cnt + start_i].idx1 = pId;
					nbq[cnt + start_i].idx2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, -1, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].constraint_type = constraint_boundary;
					nbq[cnt + start_i].inter_distance = pos[1] - hi[1];
					cnt++;
				}
				if (pos[2] > hi[2] && c3)
				{
					c3 = true;
					nbq[cnt + start_i].idx1 = pId;
					nbq[cnt + start_i].idx2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, -1);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].constraint_type = constraint_boundary;
					nbq[cnt + start_i].inter_distance = pos[2] - hi[2];
					cnt++;
				}
				if (pos[0] < lo[0] && c4)
				{
					c4 = true;
					nbq[cnt + start_i].idx1 = pId;
					nbq[cnt + start_i].idx2 = -1;
					nbq[cnt + start_i].normal1 = Coord(1, 0, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].constraint_type = constraint_boundary;
					nbq[cnt + start_i].inter_distance = lo[0] - pos[0];
					cnt++;
				}
				if (pos[1] < lo[1] && c5)
				{
					c5 = true;
					nbq[cnt + start_i].idx1 = pId;
					nbq[cnt + start_i].idx2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 1, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].constraint_type = constraint_boundary;
					nbq[cnt + start_i].inter_distance = lo[1] - pos[1];
					cnt++;
				}
				if (pos[2] < lo[2] && c6)
				{
					c6 = true;
					nbq[cnt + start_i].idx1 = pId;
					nbq[cnt + start_i].idx2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, 1);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].constraint_type = constraint_boundary;
					nbq[cnt + start_i].inter_distance = lo[2] - pos[2];
					cnt++;
				}

			}

		}
		else if (pId >= start_tet && pId < start_segment) // tets
		{
		}
		else//segments 
		{}
	}

	//template <typename Coord>
	__global__ void RB_set_friction(
		DArray<NeighborConstraints> nbq,
		int contact_size
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= contact_size) return;


		Coord3D n = nbq[pId].normal1;
		n /= n.norm();

		Coord3D n1, n2;
		if (abs(n[1]) > EPSILON || abs(n[2]) > EPSILON)
		{
			n1 = Coord3D(0, n[2], -n[1]);
			n1 /= n1.norm();
			n2 = n1.cross(n);
			n2 /= n2.norm();
		}
		else if(abs(n[0]) > EPSILON)
		{
			n1 = Coord3D(n[2], 0, -n[0]);
			n1 /= n1.norm();
			n2 = n1.cross(n);
			n2 /= n2.norm();
		}
		
		//else return;
		//printf("%d %d %d %d %.3lf %.3lf %.3lf ||| %.10lf %.10lf %.10lf \n", pId, nbq[pId].constraint_type, nbq[pId].idx1, nbq[pId].idx2, nbq[pId].normal1[0], nbq[pId].normal1[1], nbq[pId].normal1[2], n1.dot(n), n2.dot(n), n1.dot(n2));

		nbq[pId * 2 + contact_size] = nbq[pId];
		nbq[pId * 2 + contact_size].constraint_type = constraint_friction;
		nbq[pId * 2 + contact_size].normal1 = n1;
		nbq[pId * 2 + 1 + contact_size] = nbq[pId];
		nbq[pId * 2 + 1 + contact_size].constraint_type = constraint_friction;
		nbq[pId * 2 + 1 + contact_size].normal1 = n2;

		/*printf("%d %d %d %d\n", pId, 
			nbq[pId * 2 + contact_size].constraint_type, 
			nbq[pId * 2 + contact_size].idx1, 
			nbq[pId * 2 + contact_size].idx2);
*/
		
	}

		template <typename Coord>
		__global__ void RB_count_boundary(
			DArray<Sphere3D> sphere,
			DArray<Box3D> box,
			DArray<Tet3D> tet,
			DArray<int> count,
			Coord hi,
			Coord lo,
			int start_sphere,
			int start_box,
			int start_tet,
			int start_segment
		)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= sphere.size() + box.size()) return;

			if (pId < start_box && pId >= start_sphere)//sphere
			{

			}
			else if (pId >= start_box && pId < start_tet)//box
			{
				//int idx = pId - start_box;
				int cnt = 0;
//				int start_i;
				Coord center = box[pId].center;
				Coord u = box[pId].u;
				Coord v = box[pId].v;
				Coord w = box[pId].w;
				Coord extent = box[pId].extent;
				Point3D p[8];
				p[0] = Point3D(center - u * extent[0] - v * extent[1] - w * extent[2]);
				p[1] = Point3D(center - u * extent[0] - v * extent[1] + w * extent[2]);
				p[2] = Point3D(center - u * extent[0] + v * extent[1] - w * extent[2]);
				p[3] = Point3D(center - u * extent[0] + v * extent[1] + w * extent[2]);
				p[4] = Point3D(center + u * extent[0] - v * extent[1] - w * extent[2]);
				p[5] = Point3D(center + u * extent[0] - v * extent[1] + w * extent[2]);
				p[6] = Point3D(center + u * extent[0] + v * extent[1] - w * extent[2]);
				p[7] = Point3D(center + u * extent[0] + v * extent[1] + w * extent[2]);
				bool c1, c2, c3, c4, c5, c6;
				c1 = c2 = c3 = c4 = c5 = c6 = true;
				for (int i = 0; i < 8; i++)
				{
					Coord pos = p[i].origin;
					if (pos[0] > hi[0] && c1)
					{
						c1 = true;
						cnt++;
					}
					if (pos[1] > hi[1] && c2)
					{
						c2 = true;
						cnt++;
					}
					if (pos[2] > hi[2] && c3)
					{
						c3 = true;
						cnt++;
					}
					if (pos[0] < lo[0] && c4)
					{
						c4 = true;
						cnt++;
					}
					if (pos[1] < lo[1] && c5)
					{
						c5 = true;
						cnt++;
					}
					if (pos[2] < lo[2] && c6)
					{
						c6 = true;
						cnt++;
					}

				}
				count[pId] = cnt;
			}
			else if (pId >= start_tet && pId < start_segment)//tets
			{
			}
			else//segments
			{}
		}


	template<typename TDataType>
	void RigidBodySystem<TDataType>::init_boundary()
	{
		
		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(this->getTopologyModule());
		uint pDims = cudaGridSize((discreteSet->getSpheres().size() + discreteSet->getBoxes().size()), BLOCK_SIZE);
		int sum = 0;

		cnt_boudary.resize(discreteSet->getSize());
		cnt_boudary.reset();
		if (discreteSet->getSize() > 0)
		{
			printf("Yes %d\n", discreteSet->getSize());
			RB_count_boundary << <pDims, BLOCK_SIZE >> > (
				discreteSet->getSpheres(),
				discreteSet->getBoxes(),
				discreteSet->getTets(),
				cnt_boudary,
				hi,
				lo,
				start_sphere,
				start_box,
				start_tet,
				start_segment
				);
			cuSynchronize();

			sum  += m_reduce.accumulate(cnt_boudary.begin(), cnt_boudary.size());
			m_scan.exclusive(cnt_boudary, true);
			cuSynchronize();

			buffer_boundary.resize(sum);


			printf("sum = %d\n", sum);
			if (sum > 0)
				RB_set_boundary << <pDims, BLOCK_SIZE >> > (
					discreteSet->getSpheres(),
					discreteSet->getBoxes(),
					discreteSet->getTets(),
					cnt_boudary,
					buffer_boundary,
					hi,
					lo,
					start_sphere,
					start_box,
					start_tet,
					start_segment
					);
			cuSynchronize();
		}
		else
			buffer_boundary.resize(0);
		if (have_mesh_boundary)
		{

			
		}



	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::init_friction()
	{

		


	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::init_jacobi(Real dt)
	{
		//auto start = std::chrono::system_clock::now();
		
		
		if(m_shapes->getSize() > 0)
			m_nbrQueryElement->compute();
		


		

		//if (m_shapes->getSize() > 0)
			init_boundary();

			//auto end = std::chrono::system_clock::now();
			//auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			//std::cout << "oct time = " << elapsed.count() << "ms" << '\n';
		int size_constraints = buffer_boundary.size();
		
		if(m_shapes->getSize() > 0)
			size_constraints += m_nbrQueryElement->nbr_cons.getElementCount() ;
		
		printf("size constraints %d %d mshape size = %d\n", size_constraints, m_nbrQueryElement->nbr_cons.getElementCount(), m_shapes->getSize());

		

		int constraint_contact = size_constraints;
		
		if (have_friction)
		{
		
			size_constraints += 2 * buffer_boundary.size();
			if (m_shapes->getSize() > 0)
				size_constraints += 2 * m_nbrQueryElement->nbr_cons.getElementCount();

			
		}


		constraints_all.resize(size_constraints);
		
		J.resize(4 * size_constraints);
		B.resize(4 * size_constraints);
		AA.setElementCount(currentCenter()->getElementCount() * 2);
		D.resize(size_constraints);
		ita.resize(size_constraints);
		lambda.resize(size_constraints);

		J.reset();
		B.reset();
		D.reset();
		ita.reset();
		AA.getData().reset();
		lambda.reset();

		mass_eq.reset();


		if (size_constraints == 0) return;

		
		if (m_shapes->getSize() > 0  && m_nbrQueryElement->nbr_cons.getElementCount() > 0)
			cudaMemcpy(constraints_all.begin(), m_nbrQueryElement->nbr_cons.getData().begin(), m_nbrQueryElement->nbr_cons.getElementCount() * sizeof(NeighborConstraints), cudaMemcpyDeviceToDevice);
		
		
		if (buffer_boundary.size() > 0)
		{
			if (m_shapes->getSize() > 0)
			{ 
				if(!have_mesh)
					cudaMemcpy(constraints_all.begin() + m_nbrQueryElement->nbr_cons.getElementCount(), buffer_boundary.begin(), buffer_boundary.size() * sizeof(NeighborConstraints), cudaMemcpyDeviceToDevice);
				
			}
			else
			{
					cudaMemcpy(
						constraints_all.begin(),
						buffer_boundary.begin(),
						buffer_boundary.size() * sizeof(NeighborConstraints),
						cudaMemcpyDeviceToDevice
					);
			}
		}

		printf("^^^^^^^^^^^^^^^ %d %d\n", constraint_contact, constraints_all.size());

		if (have_friction)
		{
			cuExecute(constraint_contact, RB_set_friction,
				constraints_all,
				constraint_contact
				);

		}
		uint pDims = cudaGridSize(size_constraints, BLOCK_SIZE);
		uint pDimsR = cudaGridSize(currentMass()->getElementCount(), BLOCK_SIZE);


		if(use_new_mass)
		{
			printf("?????? USE NEW\n");
			mass_eq.reset();
			mass_buffer.reset();

			for (int it = 0; it < 15; it++)
			{
				mass_eq.reset();
				RB_constrct_mass_eq << <pDims, BLOCK_SIZE >> > (
					currentCenter()->getData(),
					m_inertia.getData(),
					currentMass()->getData(),
					J,
					B,
					mass_eq,
					mass_buffer,
					m_inertia.getData(),
					constraints_all
					);
				cuSynchronize();

				RB_constrct_mass_eq<< <pDimsR, BLOCK_SIZE >> > (
					m_inertia.getData(),
					currentMass()->getData(),
					mass_eq
					);
				cuSynchronize();
				//Function1Pt::copy(mass_buffer, mass_eq);
				mass_buffer.assign(mass_eq);
			}
		}
		
		RB_update_pair_info << <pDims, BLOCK_SIZE >> > (
			center_init,
			currentCenter()->getData(),
			currentRigidRotation()->getData(),
			constraints_all
			);

		cuSynchronize();
		if (use_new_mass)
		{
			RB_constrct_jacobi << <pDims, BLOCK_SIZE >> > (
				currentCenter()->getData(),
				m_inertia.getData(),
				m_inertia.getData(),
				currentMass()->getData(),
				mass_eq,
				J,
				B,
				constraints_all
				);
			cuSynchronize();
		}
		else
		{ 
			RB_constrct_jacobi << <pDims, BLOCK_SIZE >> > (
				currentCenter()->getData(),
				m_inertia.getData(),
				currentMass()->getData(),
				J,
				B,
				constraints_all
				);
			cuSynchronize();
		}
		RB_compute_d << <pDims, BLOCK_SIZE >> > (
			J,
			B,
			D
			);
		cuSynchronize();

		//printf("BEFORE ITA!!!!!\n");
		RB_compute_ita << <pDims, BLOCK_SIZE >> > (
			currentVelocity()->getData(),
			currentAngularVelocity()->getData(),
			J,
			ita,
			currentMass()->getData(),
			constraints_all,
			dt
			);
		cuSynchronize();


	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::pretreat(Real dt)
	{
		//auto start = std::chrono::system_clock::now();
		init_jacobi(dt);
		/*auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout<<"pretreat time = " << elapsed.count() << "ms" << '\n';*/
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::take_one_iteration(Real dt)
	{
		//int size_constraints = buffer_boundary.size();
		//if (have_mesh)
		//{
		//	size_constraints += m_nbrQueryMeshRigids->nbr_cons.getElementCount();
		//	//printf("????????????????????????????????????????????????????????? size of mesh constraints: %d\n", m_nbrQueryMeshRigids->nbr_cons.getElementCount());
		//}
		//if (m_nbrQueryElement != NULL)
		//{
		//	size_constraints += m_nbrQueryElement->nbr_cons.getElementCount();
		//}

		//auto start = std::chrono::system_clock::now();
		
		

		int size_constraints = constraints_all.size();
		if (size_constraints == 0) return;
		uint pDims = cudaGridSize(size_constraints, BLOCK_SIZE);

		//
		//

		RB_take_one_iteration << <pDims, BLOCK_SIZE >> > (
				AA.getData(),
				D,
				J,
				B,
				ita,
				lambda,
				currentMass()->getData(),
				constraints_all
				);
		cuSynchronize();
		
		//auto end = std::chrono::system_clock::now();
		//auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		//std::cout <<"one iteration time = " << elapsed.count() << "ms" << '\n';

	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::update_state(Real dt)
	{
		//auto start = std::chrono::system_clock::now();

		update_position_rotation(dt);
		rigid_update_topology();

		//auto end = std::chrono::system_clock::now();
		//auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		//std::cout <<"update time = " << elapsed.count() << "ms" << '\n';

	}
	template<typename TDataType>
	void RigidBodySystem<TDataType>::updateStates()
	{

		this->graphicsPipeline()->update();
		printf("inside\n");
		Real dt = Real(0.001);
		//if (this->getParent() != NULL)
		dt = this->getDt();
		//construct j
		pretreat(dt);
		for (int i = 0; i < 15; i++)
			take_one_iteration(dt);
		update_state(dt);
		

		rigid_update_topology();
		return;
		/*
		init_jacobi(dt);

		//solve_constraint 
		solve_constraint();

		update_position_rotation(dt);
		rigid_update_topology();
		*/
		
	}
	DEFINE_CLASS(RigidBodySystem);
}