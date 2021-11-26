#include "SceneGraph.h"
//#include "Action/ActDraw.h"
#include "Action/ActReset.h"
#include "Action/ActPostProcessing.h"
#include "SceneLoaderFactory.h"

namespace dyno
{
	SceneGraph& SceneGraph::getInstance()
	{
		static SceneGraph m_instance;
		return m_instance;
	}

	bool SceneGraph::isIntervalAdaptive()
	{
		return mAdvativeInterval;
	}

	void SceneGraph::setAdaptiveInterval(bool adaptive)
	{
		mAdvativeInterval = adaptive;
	}

	void SceneGraph::setGravity(Vec3f g)
	{
		mGravity = g;
	}

	Vec3f SceneGraph::getGravity()
	{
		return mGravity;
	}

	bool SceneGraph::initialize()
	{
		if (mInitialized)
		{
			return true;
		}
		//TODO: check initialization
		if (mRoot == nullptr)
		{
			return false;
		}

		class InitAct : public Action
		{
		public:
			void process(Node* node) override {
				node->initialize();

				auto& list = node->getModuleList();
				std::list<std::shared_ptr<Module>>::iterator iter = list.begin();
				for (; iter != list.end(); iter++)
				{
					(*iter)->initialize();
				}
				
				node->graphicsPipeline()->update();
			}
		};

		mRoot->traverseBottomUp<InitAct>();
		mInitialized = true;

		return mInitialized;
	}

	void SceneGraph::invalid()
	{
		mInitialized = false;
	}

	void SceneGraph::advance(float dt)
	{
		class AdvanceAct : public Action
		{
		public:
			AdvanceAct(float dt, float t) {
				mDt = dt; 
				mElapsedTime = t;
			};

			void start(Node* node) override {
				if (node == NULL)
					return;

				node->varTimeStep()->setValue(mDt);
				node->varElapsedTime()->setValue(mElapsedTime);
			}

			void process(Node* node) override {
				if (node == NULL)
				{
					Log::sendMessage(Log::Error, "Node is invalid!");
					return;
				}
				if (node->isActive())
				{
					auto customModules = node->getCustomModuleList();
					for (std::list<std::shared_ptr<CustomModule>>::iterator iter = customModules.begin(); iter != customModules.end(); iter++)
					{
						(*iter)->update();
					}

					node->update();
				}
			}

			float mDt;
			float mElapsedTime;
		};	

		mRoot->traverseTopDown<AdvanceAct>(dt, mElapsedTime);
		mElapsedTime += dt;
	}

	void SceneGraph::takeOneFrame()
	{
		std::cout << "****************Frame " << mFrameNumber << " Started" << std::endl;

		if (mRoot == nullptr)
		{
			return;
		}

		float t = 0.0f;
		float dt = 0.0f;

		class QueryTimeStep : public Action
		{
		public:
			void process(Node* node) override {
				dt = std::min(node->getDt(), dt);
			}

			float dt;
		} timeStep;

		timeStep.dt = 1.0f / mFrameRate;

		mRoot->traverseTopDown(&timeStep);
		dt = timeStep.dt;

		if (mAdvativeInterval)
		{
			this->advance(dt);
		}
		else
		{
			float interval = 1.0f / mFrameRate;
			while (t + dt < interval)
			{
				this->advance(dt);

				t += dt;
				timeStep.dt = 1.0f / mFrameRate;
				mRoot->traverseTopDown(&timeStep);
				dt = timeStep.dt;
			}

			this->advance(interval - t);
		}

// 		class UpdateGrpahicsContextAct : public Action
// 		{
// 		public:
// 			void process(Node* node) override {
// 				node->graphicsPipeline()->update();
// 			}
// 		};
// 
// 		m_root->traverseTopDown<UpdateGrpahicsContextAct>();

		mRoot->traverseTopDown<PostProcessing>();

		std::cout << "****************Frame " << mFrameNumber << " Ended" << std::endl << std::endl;

		mFrameNumber++;
	}

	void SceneGraph::updateGraphicsContext()
	{
		class UpdateGrpahicsContextAct : public Action
		{
		public:
			void process(Node* node) override {
				node->graphicsPipeline()->update();
			}
		};

		mRoot->traverseTopDown<UpdateGrpahicsContextAct>();
	}

	void SceneGraph::run()
	{

	}

	void SceneGraph::reset()
	{
		if (mRoot == nullptr)
		{
			return;
		}

		mRoot->traverseBottomUp<ResetAct>();

		//m_root->traverseBottomUp();
	}

	bool SceneGraph::load(std::string name)
	{
		SceneLoader* loader = SceneLoaderFactory::getInstance().getEntryByFileName(name);
		if (loader)
		{
			mRoot = loader->load(name);
			return true;
		}

		return false;
	}

	Vec3f SceneGraph::getLowerBound()
	{
		return mLowerBound;
	}

	Vec3f SceneGraph::getUpperBound()
	{
		return mUpperBound;
	}

	void SceneGraph::setLowerBound(Vec3f lowerBound)
	{
		mLowerBound = lowerBound;
	}

	void SceneGraph::setUpperBound(Vec3f upperBound)
	{
		mUpperBound = upperBound;
	}
}