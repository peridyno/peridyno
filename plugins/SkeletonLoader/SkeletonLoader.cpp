#include "SkeletonLoader.h"

#define AXIS 0

namespace dyno
{
	IMPLEMENT_TCLASS(SkeletonLoader, TDataType)

	template<typename TDataType>
	SkeletonLoader<TDataType>::SkeletonLoader()
		: Node()
	{
		auto defaultTopo = std::make_shared<DiscreteElements<TDataType>>();
		this->stateTopology()->setDataPtr(defaultTopo);
	}

	template<typename TDataType>
	SkeletonLoader<TDataType>::~SkeletonLoader()
	{
		
	}

	template<typename TDataType>
	bool SkeletonLoader<TDataType>::initFBX(const char* filepath)
	{
		FILE* fp = fopen(filepath, "rb");

		if (!fp) return false;

		fseek(fp, 0, SEEK_END);
		long file_size = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		auto* content = new ofbx::u8[file_size];
		fread(content, 1, file_size, fp);

		this->g_scene = ofbx::load((ofbx::u8*)content, file_size, (ofbx::u64)ofbx::LoadFlags::TRIANGULATE);

		delete[] content;
		fclose(fp);

		return true;		
	}


	template<typename TDataType>
	void SkeletonLoader<TDataType>::getAnimationCurve(const ofbx::Object& object, std::shared_ptr<JointTree<TDataType>> parent)
	{
		if (object.getType() != ofbx::Object::Type::ANIMATION_CURVE_NODE) return;
		if (strlen(object.name) != 1) return;
		auto AnimObject = (ofbx::AnimationCurveNode*)&object;
		Real d[3];
		d[0] = AnimObject->getAnimationDX();
		d[1] = AnimObject->getAnimationDY();
		d[2] = AnimObject->getAnimationDZ();
		auto curve0 = AnimObject->getCurve(0);
		int key_allsize = (curve0 == nullptr)? 1: curve0->getKeyCount();
		auto animCurve = std::make_shared<dyno::AnimationCurve<TDataType>>(key_allsize, d[0], d[1], d[2]);

		for (int i = 0; i < 3; ++i)
		{
			std::vector<long long> times;
			std::vector<float> values;
			
			auto curve = AnimObject->getCurve(i);
			if (curve == nullptr)
			{
				times.push_back(0);
				values.push_back(d[i]);
			}
			else {
				int key_count = curve->getKeyCount();

				//assert(key_allsize == key_count);

				const long long* t = curve->getKeyTime();
				const float* v = curve->getKeyValue();
				times.assign(t, t + key_count);
				values.assign(v, v + key_count);
			}

			animCurve->set(i, times, values);
		}

		switch (object.name[0])
		{
		case 'T':
			parent->setAnimTranslation(animCurve);
			break;
		case 'R':
			parent->setAnimRotation(animCurve);
			break;
		case 'S':
			parent->setAnimScaling(animCurve);
			break;		
		default:
			break;
		}
	}


	template<typename TDataType>
	void SkeletonLoader<TDataType>::getModelProperties(const ofbx::Object& object, std::shared_ptr<JointTree<TDataType>> cur)
	{
		cur->id = object.id;

		//FIXME Pre可能出错了
		copyVecR(cur->PreRotation, object.getPreRotation());
		copyVecT(cur->LclTranslation, object.getLocalTranslation());
		copyVecR(cur->LclRotation, object.getLocalRotation());
		copyVec(cur->LclScaling, object.getLocalScaling());
		cur->CurTranslation = cur->LclTranslation;
		cur->CurRotation = cur->LclRotation;
		cur->CurScaling = cur->LclScaling;

		//DFS 序
		m_jointMap.push_back(cur);
	}	

	template<typename TDataType>
	void SkeletonLoader<TDataType>::getLimbNode(const ofbx::Object& object, std::shared_ptr<JointTree<TDataType>> parent)
	{
		if (object.getType() != ofbx::Object::Type::LIMB_NODE) return;

		std::shared_ptr<JointTree<TDataType>> cur;

		if (object.getType() == ofbx::Object::Type::LIMB_NODE){

			cur = std::make_shared<JointTree<TDataType>>();
			if(parent != nullptr) parent->children.push_back(cur);
			cur->parent = parent;
			getModelProperties(object, cur);
		}
		int i = 0;
		while (ofbx::Object* child = object.resolveObjectLink(i))
		{
			if (object.getType() == ofbx::Object::Type::LIMB_NODE) 
			{
				getLimbNode(*child, cur);
				// animation curve node
				getAnimationCurve(*child, cur);
			}
			else getLimbNode(*child, parent);
			++i;
		}	
	}
	template<typename TDataType>
	void SkeletonLoader<TDataType>::getNodes(const ofbx::IScene& scene)
	{
		const ofbx::Object* root = scene.getRoot();
		if (root) {
			int i = 0;
			while (ofbx::Object* child = root->resolveObjectLink(i))
			{
				getLimbNode(*child, nullptr);
				++i;
			}			
		}		
	}

	template<typename TDataType>
	void SkeletonLoader<TDataType>::loadFBX()
	{
		auto filename = this->varFileName()->getData();
		std::string filepath = filename.string();
		m_jointMap.clear();
		initFBX(filepath.c_str());
		getNodes(*g_scene);
	}

	// 以重心为坐标原点的旋转、平移四元数转换
	template<typename TDataType>
	void SkeletonLoader<TDataType>::getCenterQuat(Coord v0, Coord v1, Quat<Real> &T, Quat<Real> &R)
	{
		Coord center = (v0 + v1) / 2.f;
		Coord tmp = v1 - v0;
		Vec3f dir = Vec3f(tmp[0], tmp[1], tmp[2]).normalize();

		float cos2;
		Vec3f axis;
		switch (AXIS)
		{
		case 0: // X (1, 0, 0) × dir = u
			cos2 = dir.x; 
			axis = Vec3f(0, -dir.z, dir.y).normalize();
			break;
		case 1: // Y (0, 1, 0) × dir = u
			cos2 = dir.y; 
			axis = Vec3f(dir.z, 0, -dir.x).normalize();
			break;
		default: // Z (0, 0, 1) × dir = u
			cos2 = dir.z; 
			axis = Vec3f(-dir.y, dir.x, 0).normalize();
			break;					
		}
		float cos1 = sqrtf((1 + cos2) / 2.0); 
		float sin1 = sqrtf((1 - cos2) / 2.0);
		Quat<Real> q(axis.x * sin1, axis.y * sin1, axis.z * sin1, cos1);
		Quat<Real> t(center[0], center[1], center[2], 0.f);
		T = t;
		R = q;
	}

	template<typename TDataType>
	bool SkeletonLoader<TDataType>::scale(Real s)
	{
		this->m_jointMap[0]->scale(s);
		
		return true;
	}
	
	template<typename TDataType>
	bool SkeletonLoader<TDataType>::translate(Coord t)
	{
		this->m_jointMap[0]->translate(t);
		return true;
	}

	Vec3f DEBUG_T(0.25, 0.0, 0.0); // 用于平移骨架，以便对比演示
	
	template<typename TDataType>
	void SkeletonLoader<TDataType>::resetStates()
	{
		loadFBX();
		if (m_jointMap.empty())
		{
			printf("Load Skeleton failed.");
			return;
		}

        // init Bone
		{
			std::vector<Coord> v0;
			std::vector<Coord> v1;
			for (auto joint : this->m_jointMap)
			{
				joint->getGlobalQuat();
				joint->getGlobalCoord();
			}
			
			int id_joint = 0;
			int id_cap = 0;
			m_capLists.clear();
			m_T.clear();
			m_R.clear();

			for (auto joint : this->m_jointMap)
			{
				for (auto joint_son : joint->children)
				{
					m_capLists.push_back(JCapsule{id_joint, id_cap, 
													joint->GlCoord, joint_son->GlCoord});
					Vec3f t0 = joint->GlCoord;
					Vec3f t1 = joint_son->GlCoord;
					t0 += DEBUG_T;
					t1 += DEBUG_T;
					v0.push_back(t0);
					v1.push_back(t1);
																	
					Quat<Real> t, r;
					getCenterQuat(joint->GlCoord, joint_son->GlCoord, t, r);
					m_T.push_back(t);
					m_R.push_back(r);
					++id_cap;
				}
				++id_joint;
			}
			m_numCaps = id_cap;
			m_numjoints = id_joint;


			// vector -> DArray
			this->outCapsule()->allocate();
			this->outCapsule()->getData().resize(m_numCaps);
			this->outCapsule()->getData().assign(m_capLists);

			this->outTranslate()->allocate();
			this->outTranslate()->getData().resize(m_numCaps);
			this->outTranslate()->getData().assign(m_T);

			this->outRotate()->allocate();
			this->outRotate()->getData().resize(m_numCaps);
			this->outRotate()->getData().assign(m_R);

			this->outPosV()->allocate();
			this->outPosV()->getData().resize(m_numCaps);
			this->outPosV()->getData().assign(v0);

			this->outPosU()->allocate();
			this->outPosU()->getData().resize(m_numCaps);
			this->outPosU()->getData().assign(v1);
			
		}

		// Init Capsule Topology
		{
			auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->stateTopology()->getDataPtr());
			mHostCap3D.clear();
			for (auto& cap : m_capLists)
			{
				Capsule3D cap3d;
				cap3d.radius = this->varRadius()->getData();
				cap3d.segment.v0 = cap.v0;
				cap3d.segment.v1 = cap.v1;
				mHostCap3D.push_back(cap3d);
			}		

			auto& caps = topo->getCaps();
			caps.resize(mHostCap3D.size());
			caps.assign(mHostCap3D);

		}
	}

	template<typename TDataType>
	void SkeletonLoader<TDataType>::updateTopology()
	{
		if (this->m_jointMap.empty())
		{
			printf("Load Skeleton failed.");
			return;
		}		
		
        //Animation
        for (auto joint : this->m_jointMap)
        {
			// static int first = 0;
			// if(first < 7)
			{
           		joint->applyAnimationAll(this->stateElapsedTime()->getData());
				// first++;
			}
            // joint->applyAnimationAll(0.05);
        }
        
		for (auto joint : m_jointMap)
		{
			joint->getGlobalQuat();
			joint->getGlobalCoord();
		}

		// Update Bone
		{
			int index = 0;
			std::vector<Coord> v0;
			std::vector<Coord> v1;			
			for (auto joint : this->m_jointMap)
			{
				for (auto joint_son : joint->children)
				{
					m_capLists[index].v0 = joint->GlCoord;
					m_capLists[index].v1 = joint_son->GlCoord;
					Quat<Real> t, r;
					getCenterQuat(joint->GlCoord, joint_son->GlCoord, t, r);
					m_T[index] = t;
					m_R[index] = r;

					Vec3f t0 = joint->GlCoord;
					Vec3f t1 = joint_son->GlCoord;
					t0 += DEBUG_T;
					t1 += DEBUG_T;					
					v0.push_back(t0);
					v1.push_back(t1);
					
					index++;
				}
			}

			this->outCapsule()->getData().assign(m_capLists);
			this->outTranslate()->getData().assign(m_T);
			this->outRotate()->getData().assign(m_R);

			this->outPosV()->getData().assign(v0);
			this->outPosU()->getData().assign(v1);
		}
		
		// Update Capsule Topology
		{
			auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->stateTopology()->getDataPtr());
			int index = 0;
			for (auto& cap : m_capLists)
			{
				auto &cap3d = mHostCap3D[index++];
				cap3d.segment.v0 = cap.v0;
				cap3d.segment.v1 = cap.v1;
			}		

			auto& caps = topo->getCaps();
			caps.assign(mHostCap3D);
		}
    }

	DEFINE_CLASS(SkeletonLoader);
}