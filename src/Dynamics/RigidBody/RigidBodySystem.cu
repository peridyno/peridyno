#pragma once
#include "RigidBodySystem.h"
#include "Topology/DiscreteElements.h"
#include "Matrix.h"
#include "Topology/NeighborConstraints.h"



namespace dyno
{
	typedef typename TOrientedBox3D<Real> Box3D;
	typedef typename Quat<Real> TQuaternion;

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

	template <typename Coord, typename Matrix, typename TQuaternion>
	__global__ void RB_initialize_device(
		DArray<Coord> pos,
		DArray<Matrix> rotation,
		DArray<TQuaternion> rotation_q,
		DArray<Sphere3D> spheres,
		DArray<Box3D> boxes,
		int start_box,
		int start_sphere
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;
		rotation[pId] = Matrix::identityMatrix();
		
		rotation_q[pId] = TQuaternion(0, 0, 0, 1);
		

		if (pId >= start_box) pos[pId] = boxes[pId - start_box].center;
		else pos[pId] = spheres[pId - start_sphere].center;
	}

	template<typename TDataType>
	bool RigidBodySystem<TDataType>::initialize()
	{
		//todo: initialize inertial tensor
		//todo: copy from topology module
		if (late_initialize)
			return true;
		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(this->getTopologyModule());//this ???
		m_box3d_init.resize(discreteSet->getBoxes().size());
		m_sphere3d_init.resize(discreteSet->getSpheres().size());

		m_box3d_init.assign(discreteSet->getBoxes());
		m_sphere3d_init.assign(discreteSet->getSpheres());

		int size_rigids = discreteSet->getBoxes().size() + discreteSet->getSpheres().size();

		start_sphere = 0;
		start_box = discreteSet->getSpheres().size();

		
		currentRotation()->setElementCount(size_rigids);
		currentAngularVelocity()->setElementCount(size_rigids);
		currentCenter()->setElementCount(size_rigids);
		currentVelocity()->setElementCount(size_rigids);
		currentMass()->setElementCount(size_rigids);
		cnt_boudary.resize(size_rigids);

		m_rotation_q.resize(size_rigids);
		m_inertia.setElementCount(size_rigids);
		m_inertia_init.resize(size_rigids);


		uint pDimsR = cudaGridSize(size_rigids, BLOCK_SIZE);
		
		RB_initialize_device << <pDimsR, BLOCK_SIZE >> > (
			currentCenter()->getData(),
			currentRotation()->getData(),
			m_rotation_q,
			m_sphere3d_init,
			m_box3d_init,
			start_box,
			start_sphere
			);
			
		cuSynchronize();


		/*
				FOR TEST!!!
		*/
		
		// TODO: initialize pos_init
		center_init.resize(size_rigids);
		center_init.assign(currentCenter()->getData());


		
		std::vector<Coord> host_angular_velocity;
		std::vector<Coord> host_velocity;

		


		/*
		INITIALIZE VELOCITY !!!
		//quaternion in pkysika: w is scalar;
		*/

		//host_velocity.push_back(Coord(0, -0.1, 0));
		//host_velocity.push_back(Coord(0.0, 1.0, 0));

		//host_angular_velocity.push_back(Coord(0));
		//host_angular_velocity.push_back(Coord(0, M_PI - M_PI, 0));
		
		

		
		std::vector<Matrix> host_inertia_tensor;
		std::vector<Real> host_mass;
		std::vector<Coord> host_pair_point;

		//host_pair_point.push_back(Coord(0.5, 0.5, 0.5));
		//host_pair_point.push_back(Coord(0.5, 0.5, 0.5));

		//pos_init.resize(size_rigids); //!!!!!!!!
		//Function1Pt::copy(pos_init, host_pair_point);
		printf("11\n");
		
		for(uint i = 0; i < discreteSet->getBoxes().size(); i ++)
		{ 
			Real mass_i = 0.05f;

			host_mass.push_back(mass_i);

			Box3D host_box_i = discreteSet->getHostBoxes(i);

			host_inertia_tensor.push_back(
				mass_i / 12.0f *
				Matrix(
				4.0f * (host_box_i.extent[1] * host_box_i.extent[1] + host_box_i.extent[2] * host_box_i.extent[2]), 0, 0,
				0, 4.0f * (host_box_i.extent[0] * host_box_i.extent[0] + host_box_i.extent[2] * host_box_i.extent[2]), 0, 
				0, 0, 4.0f * (host_box_i.extent[1] * host_box_i.extent[1] + host_box_i.extent[0] * host_box_i.extent[0])
				)
				);
			host_velocity.push_back(Coord(0.0, 0.0, 0));
			host_angular_velocity.push_back(Coord(0));
			
		}
		

		m_inertia.getData().assign(host_inertia_tensor);
		m_inertia_init.assign(host_inertia_tensor);
		currentVelocity()->getData().assign(host_velocity);
		currentAngularVelocity()->getData().assign(host_angular_velocity);
		
		currentMass()->getData().assign(host_mass);
		
		printf("INITIALIZE NEQ\n");
		/* FOR TEST ONLY */
		if(m_nbrQueryElement == NULL)
		{ 
			m_nbrQueryElement = this->template addComputeModule<NeighborElementQuery<TDataType>>("neighborhood_rigid");
			m_nbrQueryElement->setDiscreteSet(discreteSet);
			m_nbrQueryElement->initialize();
		}
		return true;
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

	template <typename Coord, typename Matrix, typename TQuaternion>
	__global__ void RB_update_state(
		DArray<Coord> pos,
		DArray<Matrix> rotation,
		DArray<TQuaternion> rotation_q,
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

		rotation_q[pId] += dt * 0.5f * TQuaternion(angular_velocity[pId][0], angular_velocity[pId][1],angular_velocity[pId][2], 0.0f).multiply_q(rotation_q[pId]);
		rotation_q[pId] = rotation_q[pId].normalize();
		rotation[pId] = rotation_q[pId].get3x3Matrix();

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

		if (nbc[pId].constraint_type == -2) // test dist constraint
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
		else if (nbc[pId].constraint_type == 1) // contact, collision
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

		else if (nbc[pId].constraint_type == 0) // boundary
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

	}


	// ignore zeta !!!!!!
	template <typename Coord>
	__global__ void RB_compute_ita(
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> J,
		DArray<Real> ita,
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
		if (nbq[pId].constraint_type == 0 || nbq[pId].constraint_type == 1)
		{
			//printf("ita @: %.10lf %.10lf\n", ita[pId], (nbq[pId].s4) / dt);
			ita[pId] += min(nbq[pId].s4, 0.001) / dt / dt / 3.0f;
		}

		//printf("ita = %.5lf\n", ita_i);
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
			d += J[4 * pId].dot(B[4 * pId]);
			d += J[4 * pId + 1].dot(B[4 * pId + 1]);
			d += J[4 * pId + 2].dot(B[4 * pId + 2]);
			d += J[4 * pId + 3].dot(B[4 * pId + 3]);
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
			if(idx2 != -1)
			{ 
				ita_i -= J[4 * pId + 2].dot(AA[idx2 * 2]);
				ita_i -= J[4 * pId + 3].dot(AA[idx2 * 2 + 1]);
			}
		}
		
		if(d[pId] > EPSILON)
		{ 
			Real delta_lambda = ita_i / d[pId];

		//printf("delta_lambda = %.3lf\n", delta_lambda);

		if (nbq[pId].constraint_type == 1 || nbq[pId].constraint_type == 0) //	PROJECTION!!!!
		{
			Real lambda_new = lambda[pId] + delta_lambda;
			if (lambda_new < 0) lambda_new = 0;
			if (lambda_new > 10) lambda_new = 10;
			delta_lambda = lambda_new - lambda[pId];
		}

		lambda[pId] += delta_lambda;

		  
		atomicAdd(&AA[idx1 * 2][0], B[4 * pId][0] * delta_lambda);
		atomicAdd(&AA[idx1 * 2][1], B[4 * pId][1] * delta_lambda);
		atomicAdd(&AA[idx1 * 2][2], B[4 * pId][2] * delta_lambda);

		atomicAdd(&AA[idx1 * 2 + 1][0], B[4 * pId + 1][0] * delta_lambda);
		atomicAdd(&AA[idx1 * 2 + 1][1], B[4 * pId + 1][1] * delta_lambda);
		atomicAdd(&AA[idx1 * 2 + 1][2], B[4 * pId + 1][2] * delta_lambda);

		if(idx2 != -1)
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

	template <typename Coord>
	__global__ void RB_take_one_iteration_cg(
		DArray<Coord> AA,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Real> d,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> ita,
		DArray<Real> lambda,
		DArray<NeighborConstraints> nbq,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbq[pId].idx1;
		int idx2 = nbq[pId].idx2;

		Real ita_i = 0;
		{
			ita_i -= (velocity[idx1] + AA[idx1 * 2] * dt).dot(J[4 * pId]);
			ita_i -= (angular_velocity[idx1] + AA[idx1 * 2 + 1] * dt).dot(J[4 * pId + 1]);

			if (idx2 != -1)
			{
				ita_i -= (velocity[idx2] + AA[idx2 * 2] * dt).dot(J[4 * pId + 2]);
				ita_i -= (angular_velocity[idx2] + AA[idx2 * 2 + 1] * dt).dot(J[4 * pId + 3]);
			}
		}

		if (nbq[pId].constraint_type == 0 || nbq[pId].constraint_type == 1)
		{
			//printf("ita @: %.10lf %.10lf\n", ita[pId], (nbq[pId].s4) / dt);
			ita_i += min(nbq[pId].s4, 0.003) / dt / 1.0f;
		}

		if (d[pId] > EPSILON)
		{
			Real delta_lambda = ita_i * 1.0f;

			//printf("delta_lambda = %.3lf\n", delta_lambda);

			if (nbq[pId].constraint_type == 1 || nbq[pId].constraint_type == 0) //	PROJECTION!!!!
			{
				Real lambda_new = lambda[pId] + delta_lambda;
				if (lambda_new < 0) lambda_new = 0;
				delta_lambda = lambda_new - lambda[pId];
			}

			lambda[pId] += delta_lambda;


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

		if(nbq[pId].constraint_type == -2)
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
			currentRotation()->getData(),
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
			currentRotation()->getData(),
			m_rotation_q,
			currentVelocity()->getData(),
			currentAngularVelocity()->getData(),
			m_inertia.getData(),
			m_inertia_init,
			dt
			);

		cuSynchronize();


	}





	template <typename Coord>
	__global__ void RB_set_boundary(
		DArray<Sphere3D> sphere,
		DArray<Box3D> box,
		DArray<int> count,
		DArray<NeighborConstraints> nbq,
		Coord hi,
		Coord lo,
		int start_sphere,
		int start_box
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= sphere.size() + box.size()) return;
		
		if (pId < start_box && pId >= start_sphere)
		{

		}
		else if (pId >= start_box)
		{
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
					nbq[cnt + start_i].constraint_type = 0;
					nbq[cnt + start_i].s4 = pos[0] - hi[0];
					cnt++;
				}
				if (pos[1] > hi[1] && c2)
				{
					c2 = true;
					nbq[cnt + start_i].idx1 = pId;
					nbq[cnt + start_i].idx2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, -1, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].constraint_type = 0;
					nbq[cnt + start_i].s4 = pos[1] - hi[1];
					cnt++;
				}
				if (pos[2] > hi[2] && c3)
				{
					c3 = true;
					nbq[cnt + start_i].idx1 = pId;
					nbq[cnt + start_i].idx2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, -1);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].constraint_type = 0;
					nbq[cnt + start_i].s4 = pos[2] - hi[2];
					cnt++;
				}
				if (pos[0] < lo[0] && c4)
				{
					c4 = true;
					nbq[cnt + start_i].idx1 = pId;
					nbq[cnt + start_i].idx2 = -1;
					nbq[cnt + start_i].normal1 = Coord(1, 0, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].constraint_type = 0;
					nbq[cnt + start_i].s4 = lo[0] - pos[0];
					cnt++;
				}
				if (pos[1] < lo[1] && c5)
				{
					c5 = true;
					nbq[cnt + start_i].idx1 = pId;
					nbq[cnt + start_i].idx2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 1, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].constraint_type = 0;
					nbq[cnt + start_i].s4 = lo[1] - pos[1];
					cnt++;
				}
				if (pos[2] < lo[2] && c6)
				{
					c6 = true;
					nbq[cnt + start_i].idx1 = pId;
					nbq[cnt + start_i].idx2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, 1);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].constraint_type = 0;
					nbq[cnt + start_i].s4 = lo[2] - pos[2];
					cnt++;
				}

			}

		}
	}

		template <typename Coord>
		__global__ void RB_count_boundary(
			DArray<Sphere3D> sphere,
			DArray<Box3D> box,
			DArray<int> count,
			Coord hi,
			Coord lo,
			int start_sphere,
			int start_box
		)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= sphere.size() + box.size()) return;

			if (pId < start_box && pId >= start_sphere)
			{

			}
			else if (pId >= start_box)
			{
				int cnt = 0;
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
		}


	template<typename TDataType>
	void RigidBodySystem<TDataType>::init_boundary()
	{
		
		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(this->getTopologyModule());
		uint pDims = cudaGridSize((discreteSet->getSpheres().size() + discreteSet->getBoxes().size()), BLOCK_SIZE);

		RB_count_boundary << <pDims, BLOCK_SIZE >> > (
			discreteSet->getSpheres(),
			discreteSet->getBoxes(),
			cnt_boudary,
			hi,
			lo,
			start_sphere,
			start_box
			);
		cuSynchronize();

		int sum = m_reduce.accumulate(cnt_boudary.begin(), cnt_boudary.size());
		m_scan.exclusive(cnt_boudary, true);
		cuSynchronize();

		buffer_boundary.resize(sum);

		printf("initialize boundary constraint sum: %d\n", sum);
		if (sum <= 0) return;
		

		RB_set_boundary << <pDims, BLOCK_SIZE >> > (
			discreteSet->getSpheres(),
			discreteSet->getBoxes(),
			cnt_boudary,
			buffer_boundary,
			hi,
			lo,
			start_sphere,
			start_box
			);
		cuSynchronize();

	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::init_jacobi(Real dt)
	{
		m_nbrQueryElement->compute();
		init_boundary();

		int size_constraints = m_nbrQueryElement->nbr_cons.getElementCount() + buffer_boundary.size();
		
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


		if (size_constraints == 0) return;

		if (m_nbrQueryElement->nbr_cons.getElementCount() > 0)
			cudaMemcpy(constraints_all.begin(), m_nbrQueryElement->nbr_cons.getData().begin(), m_nbrQueryElement->nbr_cons.getElementCount() * sizeof(NeighborConstraints), cudaMemcpyDeviceToDevice);
		if (buffer_boundary.size() > 0)
			cudaMemcpy(constraints_all.begin() + m_nbrQueryElement->nbr_cons.getElementCount(), buffer_boundary.begin(), buffer_boundary.size() * sizeof(NeighborConstraints), cudaMemcpyDeviceToDevice);


		uint pDims = cudaGridSize(size_constraints, BLOCK_SIZE);
		RB_update_pair_info << <pDims, BLOCK_SIZE >> > (
			center_init,
			currentCenter()->getData(),
			currentRotation()->getData(),
			constraints_all
			);

		cuSynchronize();

		RB_constrct_jacobi << <pDims, BLOCK_SIZE >> > (
			currentCenter()->getData(),
			m_inertia.getData(),
			currentMass()->getData(),
			J,
			B,
			constraints_all
			);
		cuSynchronize();

		RB_compute_d << <pDims, BLOCK_SIZE >> > (
			J,
			B,
			D
			);
		cuSynchronize();

		printf("BEFORE ITA!!!!!\n");
		RB_compute_ita << <pDims, BLOCK_SIZE >> > (
			currentVelocity()->getData(),
			currentAngularVelocity()->getData(),
			J,
			ita,
			constraints_all,
			dt
			);
		cuSynchronize();


	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::pretreat(Real dt)
	{
		init_jacobi(dt);

	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::take_one_iteration(Real dt)
	{
		int size_constraints = m_nbrQueryElement->nbr_cons.getElementCount() + buffer_boundary.size();
		if (size_constraints == 0) return;
		uint pDims = cudaGridSize(size_constraints, BLOCK_SIZE);

		// todo: update AA;
		/*
		RB_take_one_iteration_cg << <pDims, BLOCK_SIZE >> > (
			AA.getData(),
			currentVelocity()->getData(),
			currentAngularVelocity()->getData(),
			D,
			J,
			B,
			ita,
			lambda,
			constraints_all,
			dt
			);
		cuSynchronize();*/
		

		RB_take_one_iteration << <pDims, BLOCK_SIZE >> > (
				AA.getData(),
				D,
				J,
				B,
				ita,
				lambda,
				constraints_all
				);
		cuSynchronize();
		
		

	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::update(Real dt)
	{
		update_position_rotation(dt);
		rigid_update_topology();

	}
	template<typename TDataType>
	void RigidBodySystem<TDataType>::advance(Real dt)
	{
		/*
		//construct j
		pretreat(dt);
		for (int i = 0; i < 15; i++)
			take_one_iteration(dt);
		update(dt);
		*/

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