//#define _CRT_SECURE_NO_WARNINGS

#include "Log.h"
#include "FilePath.h"


namespace dyno
{
	std::string Log::sOutputFile;
	std::ofstream Log::sOutputStream;
    void(*Log::receiver)(const Message&) = nullptr;
    Log::MessageType Log::sLogLevel = Log::DebugInfo;

    std::atomic<Log*> Log::sLogInstance = nullptr;
    std::queue<Log::Message> Log::sMessageQueue;

#define LOG_DEBUG(format, ...) Log::instance()->writeMessage(Log::DebugInfo, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...)  Log::instance()->writeMessage(Log::Info, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) Log::instance()->writeMessage(Log::Error, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) Log::instance()->writeMessage(Log::Warning, format, ##__VA_ARGS__)
#define LOG_USER(format, ...) Log::instance()->writeMessage(Log::User, format, ##__VA_ARGS__)
#define LOG_DISPLAY(format, ...) Log::instance()->writeMessage(Log::Display, format, ##__VA_ARGS__)

    void Log::writeMessage(MessageType level, const char* format, ...)
    {
		// log message
		Log::Message m;
		m.type = level;
		m.text = std::string(format);
		time_t t = time(NULL);
		m.when = localtime(&t);

        //Finish create Message and push into queue
        {
            std::lock_guard<std::mutex> lock(mtx);
            sMessageQueue.push(m);  //push message into queue
            mCondition.notify_one();
        }
    }

    void Log::sendMessage(MessageType type, const std::string& text)
    {
		// Skip logging to file if minimum level is higher
		if ((int)type < (int)sLogLevel)
			return;

        switch (type)
        {
        case DebugInfo: LOG_DEBUG(text.c_str()); break;
        case Info:		LOG_INFO(text.c_str()); break;
        case Warning:	LOG_WARN(text.c_str()); break;
        case Error:		LOG_ERROR(text.c_str()); break;
        default:		LOG_USER(text.c_str());
        }
    }

    void Log::setUserReceiver(void (*userFunc)(const Message&))
    {
        receiver = userFunc;
    }

	void Log::setLevel(MessageType level)
	{
        sLogLevel = level;
	}

	void Log::setOutput(const std::string& filename)
	{
		sOutputFile = filename;

		// close old one
		if (sOutputStream.is_open())
			sOutputStream.close();

		// create file
		sOutputStream.open(filename.c_str());
		if (!sOutputStream.is_open())
			sendMessage(Error, "Cannot create/open '" + filename + "' for logging");
	}

	const std::string& Log::getOutput()
	{
        return sOutputFile;
	}

	Log* Log::instance()
    {
        static std::mutex mutex;
        Log* ins = sLogInstance.load(std::memory_order_acquire);

        if (!ins) {
            std::lock_guard<std::mutex> tLock(mutex);
            ins = sLogInstance.load(std::memory_order_relaxed);
            if (!ins) {
                ins = new Log();
                sLogInstance.store(ins, std::memory_order_release);
            }
        }

        return ins;
    }

    Log::Log()
        : mRunning(true)
    {
        mThread = std::thread(&Log::outputThread, this);
    }

    Log::~Log()
    {

        mCondition.notify_one();
        if (mThread.joinable()) {
            mThread.join();
        }

        mRunning = false;
    }

    void Log::outputThread()
    {
        while (mRunning) {
            std::unique_lock<std::mutex> lock(mtx);
            mCondition.wait(lock, [&]() { return !sMessageQueue.empty() || !mRunning; });

            while (!sMessageQueue.empty()) {
                auto m = sMessageQueue.front();
                sMessageQueue.pop();

				if (receiver) {
					receiver(m);
				}

				// if enabled logging to file
				if (sOutputStream.is_open())
				{
					// print time
					char buffer[9];
					strftime(buffer, 9, "%X", m.when);
					sOutputStream << buffer;

					// print type
					switch (m.type)
					{
					case DebugInfo: sOutputStream << " | Debug   | "; break;
					case Info:		sOutputStream << " | Info    | "; break;
					case Warning:	sOutputStream << " | warning | "; break;
					case Error:		sOutputStream << " | ERROR   | "; break;
					default:		sOutputStream << " | user    | ";
					}

					// print description
					sOutputStream << m.text << std::endl;
				}
            }

        }
    }
}