#include "common/logging.h"


/**************  Common parameters and functions for log  ***************/
const char* color_start[] = {
    "\033[0;32m", // green
    "\033[0;33m", // yellow
    "\033[0;31m", // red
    "\033[0;34m", // blue
    "\033[0;31m", // red
};
const char  color_end[]     = "\033[0m";
const char* log_level_str[] = {"INFO", "WARN", "ERROR", "TEMP", "FATAL"};

void getCurrentTime(char* time_buf) {
    time_t         cur_time   = time(NULL);
    tm*            rough_time = localtime(&cur_time);
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sprintf(time_buf, "%04d-%02d-%02d %02d:%02d:%02d.%03d", rough_time->tm_year + 1900,
            rough_time->tm_mon + 1, rough_time->tm_mday, rough_time->tm_hour, rough_time->tm_min,
            rough_time->tm_sec, static_cast<int>(tv.tv_usec / 1000));
}
/**************************************************************************/
void LogSettings::SetMinimumLogLevel(uint32_t minimum_log_level) {
    set_mtx_.lock();
    allow_output_log_level_ = minimum_log_level < static_cast<uint32_t>(LOG_LEVEL_FATAL)
                              ? minimum_log_level
                              : static_cast<uint32_t>(LOG_LEVEL_FATAL);
    set_mtx_.unlock();
}

void LogSettings::SetTargetLogFile(const std::string& file_path) {
    set_mtx_.lock();
    output_destination_ = file_path;
    if (FILE* fp = fopen(output_destination_.c_str(), "w")) {
        log_to_stderr_ = false;
        fclose(fp);
    } else {
        printf("The file path not initialize, target path `%s` not exist. "
               "\033[0;33mWARNING\033[0m : Use default output path [stderr].\n",
               file_path.c_str());
    }
    set_mtx_.unlock();
}

LogStream::LogStream() : stream_buf_(buffer_, MaxBufferLen) { rdbuf(&stream_buf_); }

void LogStream::Flush() {
    FILE* fp = LogSettings::GetInstance()->IfLogToStderr()
                   ? stderr
                   : fopen(LogSettings::GetInstance()->GetDestination().c_str(), "a");
    size_t num_chars = stream_buf_.pcount();
    char*  last_char = stream_buf_.pbase() + num_chars;
    if (*(last_char - 1) != '\n') {
        *last_char = '\n';
        num_chars++;
    }
    fwrite(stream_buf_.pbase(), 1, num_chars, fp);
    if (fp == stderr) {
        std::fflush(stderr);
    } else {
        fclose(fp);
    }
}

void LogMessage::initLogStream(uint32_t level, const char* file, int line) {
    log_stream_.flush();
    char cur_time[32];
    getCurrentTime(cur_time);
    if (LogSettings::GetInstance()->IfLogToStderr()) {
        log_stream_ << color_start[level] << '[' << log_level_str[level] << ']' << color_end;
    } else {
        log_stream_ << '[' << log_level_str[level] << ']';
    }
    log_stream_ << '[' << cur_time << "][" << file << ':' << line << "] ";
}

LogMessage::LogMessage(uint32_t level) : level_(level) { initLogStream(level); }

LogMessage::LogMessage(uint32_t level, const char* file, int line) : level_(level) {
    initLogStream(level, file, line);
}

std::ostream& LogMessage::stream() { return this->log_stream_; }

LogMessage::~LogMessage() {
    if (level_ >= LogSettings::GetInstance()->GetAllowedLogLevel()) {
        log_stream_.Flush();
    }
    if (level_ >= LOG_LEVEL_FATAL) {
        printf("...exec failed...\n");
        abort();
    }
}
