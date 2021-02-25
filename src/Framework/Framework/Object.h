#pragma once
#include <string>

namespace dyno
{
class Object;
class ClassInfo;


typedef Object* (*ObjectConstructorFn)(void);
bool Register(ClassInfo* ci);

class ClassInfo
{
public:
	ClassInfo(const std::string className, ObjectConstructorFn ctor)
		:m_className(className), m_objectConstructor(ctor)
	{
		Register(this);
	}
	virtual ~ClassInfo() {}

	inline bool operator==(const ClassInfo &ci) const
	{

		if (this->m_className == ci.m_className)
			return true;
		return false;
	}

	Object* createObject()const 
	{ 
		return m_objectConstructor ? (*m_objectConstructor)() : 0; 
	}

	bool isDynamic()const 
	{ 
		return NULL != m_objectConstructor; 
	}
	const std::string getClassName()const { return m_className; }

	ObjectConstructorFn getConstructor()const { return m_objectConstructor; }
public:
	std::string m_className;
	ObjectConstructorFn m_objectConstructor;
};

#define DECLARE_CLASS(name) \
public: \
    static const ClassInfo* ms_classinfo; \
public:  \
    virtual const ClassInfo* getClassInfo() const; \
    static Object* createObject();


#define IMPLEMENT_CLASS_COMMON(name,func) \
const ClassInfo* name::ms_classinfo = new ClassInfo((#name), (ObjectConstructorFn) func); \
                        \
const ClassInfo* name::getClassInfo() const \
    {return name::ms_classinfo;}

#define IMPLEMENT_CLASS(name)            \
IMPLEMENT_CLASS_COMMON(name,name::createObject) \
Object* name::createObject()                   \
    { return new name;}


#define _STR(s) #s

#define DECLARE_CLASS_1(name, T1) \
public: \
    static const ClassInfo ms_classinfo; \
public:  \
    virtual const ClassInfo* getClassInfo() const; \
    static Object* createObject();

#define IMPLEMENT_CLASS_COMMON_1(name, T1, func) \
template<typename T1>		\
const ClassInfo name<T1>::ms_classinfo(std::string(_STR(name))+std::string("<")+std::string(T1::getName())+std::string(">"), \
            (ObjectConstructorFn) func); \
							\
template<typename T1>		\
const ClassInfo* name<T1>::getClassInfo() const \
    {return &name<T1>::ms_classinfo;}

#define IMPLEMENT_CLASS_1(name, T1)            \
IMPLEMENT_CLASS_COMMON_1(name, T1, name<T1>::createObject) \
							\
template<typename T1>		\
Object* name<T1>::createObject()                   \
    { return new name<T1>;}

class Object
{
	DECLARE_CLASS(Object)
public:
	Object() {};
	virtual ~Object() {};
	static bool registerClass(ClassInfo* ci);
	static Object* createObject(std::string name);
	static std::map< std::string, ClassInfo*>* getClassMap();
};


}