#pragma once

#include <string.h>
#include <sys/time.h>

#include <ctime>
#include <exception>
#include <iostream>
#include <mutex>
#include <sstream>

#include "common/singleton.h"

enum { LOG_LEVEL_INFO = 0, LOG_LEVEL_WARN = 1, LOG_LEVEL_ERROR = 2, LOG_LEVEL_TEMP = 3, LOG_LEVEL_FATAL = 4 };
class LogMessage;
class LogSettings;
class DummyLogDecorator;

#define SET_MINIMUM_LOG_LEVEL(log_level) \
    LogSettings::GetInstance()->SetMinimumLogLevel(static_cast<uint32_t>(log_level))
#define SET_OUTPUT_LOG_FILE(output_file) LogSettings::GetInstance()->SetTargetLogFile(output_file)

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define LogMessageHeader(level) LogMessage(LOG_LEVEL_##level, __FILENAME__, __LINE__)

#define LOG_IF(level, condition) !(condition) ? (void)0 : DummyLogDecorator() & LogMessageHeader(level).stream()

#define LOG(level) LOG_IF(level, true)

#ifndef NDEBUG
#define DLOG(level) LOG_IF(level, true)
#else
#define DLOG(level) LOG_IF(level, false)
#endif

#define report_error(...) print_error_information(__VA_ARGS__);

template <typename... Args> void print_error_information(Args... args) {
    std::ostringstream oss;
    ((oss << args), ...);
    printf("\033[0;31m[ERROR]\033[0m %s\n", oss.str().c_str());
    throw std::runtime_error("Terminate due to error.");
}

#define REPORT_ERROR_IF(condition, ...) \
    do {                                \
        if ((condition)) {              \
            report_error(__VA_ARGS__);  \
        }                               \
    } while (0)

/*************************************************************************************
 * Necessary Log class definition
 *  LogSettings    : Global setting for log
 *  LogStreamBuffer: stream buffer in memory, provide interface to process buffer
 *  LogStream      : Log core implementation
 *  LogMessage     : Main process of log, including preprocess and postprocess, etc.
 */
class LogSettings : public Singleton<LogSettings> {
 public:
    friend class Singleton<LogSettings>;

    void SetMinimumLogLevel(uint32_t minimum_log_level = 0);
    void SetTargetLogFile(const std::string& file_path = "");

    bool               IfLogToStderr() const { return log_to_stderr_; }
    const std::string& GetDestination() { return output_destination_; }
    uint32_t           GetAllowedLogLevel() const { return allow_output_log_level_; }

 private:
    LogSettings()                       = default;
    uint32_t    allow_output_log_level_ = 0;
    bool        log_to_stderr_          = true;
    std::string output_destination_     = "";
    std::mutex  set_mtx_;
};

class LogStreamBuffer : public std::streambuf {
 public:
    LogStreamBuffer(char* buf_begin, int len) { setp(buf_begin, buf_begin + len - 2); }

    size_t pcount() const { return pptr() - pbase(); }
    char*  pbase() const { return std::streambuf::pbase(); }
};

constexpr int MaxBufferLen = 1024;

class LogStream : public std::ostream {
 public:
    LogStream();
    void Flush();

 private:
    char            buffer_[MaxBufferLen];
    LogStreamBuffer stream_buf_;
    // Avoid copy construct.
    LogStream& operator=(const LogStream&);
    LogStream(const LogStream&);
};

class LogMessage {
 public:
    LogMessage(uint32_t level);

    LogMessage(uint32_t level, const char* file, int line);

    void initLogStream(uint32_t level, const char* file = nullptr, int line = -1);

    std::ostream& stream();

    ~LogMessage();

 private:
    uint32_t  level_;
    LogStream log_stream_;
};

class DummyLogDecorator {
 public:
    void operator&(const std::ostream&) {}
};
