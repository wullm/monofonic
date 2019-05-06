#pragma once

#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <fstream>
#include <iostream>

namespace csoca {

enum LogLevel : int {
  Off     = 0,
  Fatal   = 1,
  Error   = 2,
  Warning = 3,
  Info    = 4,
  Debug   = 5
};

class Logger {
private:
  static LogLevel log_level_;
  static std::ofstream output_file_;

public:
  Logger()  = default;
  ~Logger() = default;

  static void SetLevel(const LogLevel &level);
  static LogLevel GetLevel();

  static void SetOutput(const std::string filename);
  static void UnsetOutput();

  static std::ofstream &GetOutput();

  template <typename T> Logger &operator<<(const T &item) {
    std::cout << item;
    if (output_file_.is_open()) {
      output_file_ << item;
    }
    return *this;
  }

  Logger &operator<<(std::ostream &(*fp)(std::ostream &)) {
    std::cout << fp;
    if (output_file_.is_open()) {
      output_file_ << fp;
    }
    return *this;
  }
};

class LogStream {
private:
  Logger &logger_;
  LogLevel stream_level_;
  std::string line_prefix_, line_postfix_;

  bool newline;

public:
  LogStream(Logger &logger, const LogLevel &level)
    : logger_(logger), stream_level_(level), newline(true) {
    switch (stream_level_) {
      case LogLevel::Fatal:
        line_prefix_ = "\033[31mFatal : ";
        break;
      case LogLevel::Error:
        line_prefix_ = "\033[31mError : ";
        break;
      case LogLevel::Warning:
        line_prefix_ = "\033[33mWarning : ";
        break;
      case LogLevel::Info:
        //line_prefix_ = " | Info    | ";
        line_prefix_ = " \033[0m";
        break;
      case LogLevel::Debug:
        line_prefix_ = "Debug : \033[0m";
        break;
      default:
        line_prefix_ = "\033[0m";
        break;
    }
    line_postfix_ = "\033[0m";
  }
  ~LogStream() = default;

  inline std::string GetPrefix() const {
    return line_prefix_;
  }

  template <typename T> LogStream &operator<<(const T &item) {
    if (Logger::GetLevel() >= stream_level_) {
      if (newline) {
        logger_ << line_prefix_;
        newline = false;
      }
      logger_ << item;
    }
    return *this;
  }

  LogStream &operator<<(std::ostream &(*fp)(std::ostream &)) {
    if (Logger::GetLevel() >= stream_level_) {
      logger_ << fp;
      logger_ << line_postfix_;
      newline = true;
    }
    return *this;
  }

  inline void Print(const char *str, ...) {
    char out[1024];
    va_list argptr;
    va_start(argptr, str);
    vsprintf(out, str, argptr);
    va_end(argptr);
    std::string out_string = std::string(out);
    out_string.erase(std::remove(out_string.begin(), out_string.end(), '\n'),
                     out_string.end());
    (*this) << out_string << std::endl;
  }
};

// global instantiations for different levels
extern Logger glogger;
extern LogStream flog;
extern LogStream elog;
extern LogStream wlog;
extern LogStream ilog;
extern LogStream dlog;

} // namespace csoca
