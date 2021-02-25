#pragma once
#include "Base.h"
#include "Node.h"
#include "NodeIterator.h"

namespace dyno {


class SceneGraph : public Base
{
public:
	typedef NodeIterator Iterator;

	~SceneGraph() {};

	void setRootNode(std::shared_ptr<Node> root) { m_root = root; }
	std::shared_ptr<Node> getRootNode() { return m_root; }

	virtual bool initialize();
	bool isInitialized() { return m_initialized; }
	void invalid();

	virtual void draw();
	virtual void advance(float dt);
	virtual void takeOneFrame();
	virtual void run();

	void reset();

	virtual bool load(std::string name);

	virtual void invoke(unsigned char type, unsigned char key, int x, int y) {};

	template<class TNode, class ...Args>
	std::shared_ptr<TNode> createNewScene(Args&& ... args)
	{
		std::shared_ptr<TNode> root = TypeInfo::New<TNode>(std::forward<Args>(args)...);
		m_root = root;
		return root;
	}

public:
	static SceneGraph& getInstance();

	inline void setTotalTime(float t) { m_maxTime = t; }
	inline float getTotalTime() { return m_maxTime; }

	inline void setFrameRate(float frameRate) { m_frameRate = frameRate; }
	inline float getFrameRate() { return m_frameRate; }
	inline float getTimeCostPerFrame() { return m_frameCost; }
	inline float getFrameInterval() { return 1.0f / m_frameRate; }
	inline int getFrameNumber() { return m_frameNumber; }

	bool isIntervalAdaptive();
	void setAdaptiveInterval(bool adaptive);

	void setGravity(Vector3f g);
	Vector3f getGravity();

	Vector3f getLowerBound();
	Vector3f getUpperBound();

	void setLowerBound(Vector3f lowerBound);
	void setUpperBound(Vector3f upperBound);

	inline Iterator begin() { return NodeIterator(m_root); }
	inline Iterator end() {return NodeIterator(nullptr);	}

public:
	SceneGraph()
		: m_elapsedTime(0)
		, m_maxTime(0)
		, m_frameRate(25)
		, m_frameNumber(0)
		, m_frameCost(0)
		, m_initialized(false)
		, m_lowerBound(0, 0, 0)
		, m_upperBound(1, 1, 1)
	{
		m_gravity = Vector3f(0.0f, -9.8f, 0.0f);
	};

	/**
	* To avoid erroneous operations
	*/
	SceneGraph(const SceneGraph&) {};
	SceneGraph& operator=(const SceneGraph&) {};

private:
	bool m_initialized;
	bool m_advative_interval = true;

	float m_elapsedTime;
	float m_maxTime;
	float m_frameRate;
	float m_frameCost;

	int m_frameNumber;

	Vector3f m_gravity;

	Vector3f m_lowerBound;
	Vector3f m_upperBound;

private:
	std::shared_ptr<Node> m_root = nullptr;
};

}